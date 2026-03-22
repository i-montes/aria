use crate::vector::index::{IndexError, Result, SearchResult, VectorIndex};
use dashmap::{DashMap, DashSet};
use hnsw::{Hnsw, Searcher};
use space::{MetricPoint, Neighbor};
use std::sync::RwLock;
use rand_pcg::Pcg64;

const SCALE_FACTOR: f32 = 1_000_000_000.0;

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
        
        // Use 9 decimals of precision mapped to u32
        (distance * SCALE_FACTOR) as u32
    }
}

pub struct HnswIndex {
    dimension: usize,
    // Use standard defaults (M=12, M0=24) which are more robust in this hnsw crate version
    hnsw: RwLock<Hnsw<VectorPoint, Pcg64, 12, 24>>,
    id_to_uuid: DashMap<usize, String>,
    uuid_to_id: DashMap<String, usize>,
    deleted_internal_ids: DashSet<usize>,
}

impl HnswIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            hnsw: RwLock::new(Hnsw::new()),
            id_to_uuid: DashMap::new(),
            uuid_to_id: DashMap::new(),
            deleted_internal_ids: DashSet::new(),
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(&self, id: String, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(IndexError::Index(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        let mut hnsw = self.hnsw.write().unwrap();
        let mut searcher = Searcher::default();
        
        let internal_id = hnsw.insert(VectorPoint(vector.to_vec()), &mut searcher);
        
        self.uuid_to_id.insert(id.clone(), internal_id);
        self.id_to_uuid.insert(internal_id, id);

        Ok(())
    }

    fn remove(&self, id: String) -> Result<()> {
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
        let mut searcher = Searcher::default();
        
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
            
            // IMPORTANT: In hnsw 0.10.x, if ef >= node_count, the search might return 
            // fewer candidates than ef, but the crate attempts to copy ef elements 
            // from an internal slice that might have even fewer elements (node_count-1 or similar).
            // This causes a panic in copy_from_slice.
            // For small collections (< 100), we stay well below node_count using division.
            // For larger collections, node_count - 1 is generally safe.
            let search_ef = if node_count < 100 {
                ef.min(node_count / 2).max(1)
            } else {
                ef.min(node_count - 1).max(1)
            };
            
            let mut neighbors = vec![Neighbor { index: 0, distance: 0 }; search_ef];
            
            // Note: the 2nd parameter in nearest() is 'ef' (search depth)
            let found_neighbors = hnsw.nearest(&VectorPoint(query.to_vec()), search_ef, &mut searcher, &mut neighbors);

            results.clear();
            for neighbor in found_neighbors {
                if self.deleted_internal_ids.contains(&neighbor.index) {
                    continue;
                }

                if let Some(uuid) = self.id_to_uuid.get(&neighbor.index) {
                    let similarity = 1.0 - (neighbor.distance as f32 / SCALE_FACTOR);
                    results.push(SearchResult {
                        id: uuid.value().clone(),
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
    use crate::core::generate_id;

    #[test]
    fn test_hnsw_minimal() {
        let index = HnswIndex::new(3);
        let id1 = generate_id();
        index.add(id1.clone(), &[1.0, 0.0, 0.0]).unwrap();
        
        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert!(!results.is_empty(), "Results should not be empty for a single node");
        assert_eq!(results[0].id, id1);
    }

    #[test]
    fn test_hnsw_tombstones() {
        let index = HnswIndex::new(3);
        
        let id1 = generate_id();
        let id2 = generate_id();

        index.add(id1.clone(), &[1.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), &[0.0, 1.0, 0.0]).unwrap();

        // Add dummy nodes
        for i in 0..50 {
            index.add(generate_id(), &[0.1, 0.1, i as f32 / 50.0]).unwrap();
        }

        // Search for id1
        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.iter().any(|r| r.id == id1));

        // Remove id1
        index.remove(id1.clone()).unwrap();

        // Search again for id1's location
        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.iter().any(|r| r.id == id1), "id1 should be tombstoned");

        // Search for id2's location
        let results = index.search(&[0.0, 1.0, 0.0], 5).unwrap();
        assert!(results.iter().any(|r| r.id == id2), "id2 should be found when searching for its coordinates");
    }
}
