use crate::vector::index::{IndexError, Result, SearchResult, VectorIndex};
use dashmap::{DashMap, DashSet};
use hnsw::{Hnsw, Searcher};
use space::{MetricPoint, Neighbor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use uuid::Uuid;
use rand_pcg::Pcg64;

#[derive(Clone, Debug)]
pub struct VectorPoint(pub Vec<f32>);

impl MetricPoint for VectorPoint {
    type Metric = u32;

    fn distance(&self, other: &Self) -> Self::Metric {
        let a = &self.0;
        let b = &other.0;
        
        let mut dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot /= norm_a * norm_b;
        } else {
            dot = 0.0;
        }

        let similarity = dot.clamp(-1.0, 1.0);
        let distance = 1.0 - similarity;
        
        // Precision of 5 decimals mapped to u32
        (distance * 100_000.0) as u32
    }
}

pub struct HnswIndex {
    dimension: usize,
    // M=32 for better connectivity in high dimensions
    // M0=64 for robust base layer
    hnsw: RwLock<Hnsw<VectorPoint, Pcg64, 32, 64>>,
    searcher: RwLock<Searcher<u32>>,
    id_to_uuid: DashMap<usize, Uuid>,
    uuid_to_id: DashMap<Uuid, usize>,
    deleted_internal_ids: DashSet<usize>,
    next_id: AtomicUsize,
}

impl HnswIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            hnsw: RwLock::new(Hnsw::new()),
            searcher: RwLock::new(Searcher::default()),
            id_to_uuid: DashMap::new(),
            uuid_to_id: DashMap::new(),
            deleted_internal_ids: DashSet::new(),
            next_id: AtomicUsize::new(0),
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(&self, id: Uuid, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(IndexError::Index(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        let internal_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.uuid_to_id.insert(id, internal_id);
        self.id_to_uuid.insert(internal_id, id);

        let mut hnsw = self.hnsw.write().unwrap();
        let mut searcher = self.searcher.write().unwrap();
        
        hnsw.insert(VectorPoint(vector.to_vec()), &mut searcher);

        Ok(())
    }

    fn remove(&self, id: Uuid) -> Result<()> {
        if let Some((_, internal_id)) = self.uuid_to_id.remove(&id) {
            self.id_to_uuid.remove(&internal_id);
            self.deleted_internal_ids.insert(internal_id);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(IndexError::Index("Query dimension mismatch".to_string()));
        }

        let hnsw = self.hnsw.read().unwrap();
        let mut searcher = self.searcher.write().unwrap();
        
        let node_count = self.id_to_uuid.len() + self.deleted_internal_ids.len();
        if node_count == 0 {
            return Ok(Vec::new());
        }

        // Variable ef_search: starts at 40, scales with k, capped at 250
        let mut ef = (k * 4).clamp(40, 250);
        let mut results = Vec::new();
        let mut attempts = 0;

        // Loop to handle tombstones: if we don't find enough valid results, 
        // we expand the search area (ef) up to the limit.
        while results.len() < k && attempts < 3 {
            attempts += 1;
            let actual_neighbors_to_fetch = ef.min(node_count).max(1);
            let mut neighbors = vec![Neighbor { index: 0, distance: 0 }; actual_neighbors_to_fetch];
            
            // Note: the 2nd parameter in nearest() is 'ef' (search depth)
            let found_neighbors = hnsw.nearest(&VectorPoint(query.to_vec()), ef, &mut searcher, &mut neighbors);

            results.clear();
            for neighbor in found_neighbors {
                if self.deleted_internal_ids.contains(&neighbor.index) {
                    continue;
                }

                if let Some(uuid) = self.id_to_uuid.get(&neighbor.index) {
                    let similarity = 1.0 - (neighbor.distance as f32 / 100_000.0);
                    results.push(SearchResult {
                        id: *uuid.value(),
                        score: similarity,
                    });
                }
            }

            if results.len() >= k || ef >= 250 || ef >= node_count {
                break;
            }

            // Expand search scope if tombstones blocked our results
            ef = (ef * 2).min(250);
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_minimal() {
        let index = HnswIndex::new(3);
        let id1 = Uuid::new_v4();
        index.add(id1, &[1.0, 0.0, 0.0]).unwrap();
        
        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert!(!results.is_empty(), "Results should not be empty for a single node");
        assert_eq!(results[0].id, id1);
    }

    #[test]
    fn test_hnsw_tombstones() {
        let index = HnswIndex::new(3);
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index.add(id1, &[1.0, 0.0, 0.0]).unwrap();
        index.add(id2, &[0.0, 1.0, 0.0]).unwrap();

        // Add dummy nodes
        for i in 0..50 {
            index.add(Uuid::new_v4(), &[0.1, 0.1, i as f32 / 50.0]).unwrap();
        }

        // Search for id1
        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.iter().any(|r| r.id == id1));

        // Remove id1
        index.remove(id1).unwrap();

        // Search again for id1's location
        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.iter().any(|r| r.id == id1), "id1 should be tombstoned");

        // Search for id2's location
        let results = index.search(&[0.0, 1.0, 0.0], 5).unwrap();
        assert!(results.iter().any(|r| r.id == id2), "id2 should be found when searching for its coordinates");
    }
}
