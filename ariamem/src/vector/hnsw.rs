use crate::vector::index::{IndexError, Result, SearchResult, VectorIndex};
use dashmap::{DashMap, DashSet};
use hnsw::{Hnsw, Searcher};
use space::{MetricPoint, Neighbor};
use std::sync::RwLock;
use rand_pcg::Pcg64;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::fs::File;
use ndarray::ArrayView1;

const SCALE_FACTOR: f32 = 1_000_000_000.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorPoint(pub Vec<f32>);

impl MetricPoint for VectorPoint {
    type Metric = u32;

    fn distance(&self, other: &Self) -> Self::Metric {
        let a = &self.0;
        let b = &other.0;
        
        if a.is_empty() || b.is_empty() {
            return SCALE_FACTOR as u32;
        }

        let av = ArrayView1::from(a);
        let bv = ArrayView1::from(b);

        let mut dot = av.dot(&bv);
        let norm_a = av.dot(&av).sqrt();
        let norm_b = bv.dot(&bv).sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot /= norm_a * norm_b;
        } else {
            dot = 0.0;
        }

        let similarity = dot.clamp(-1.0, 1.0);
        let distance = 1.0 - similarity;
        
        (distance * SCALE_FACTOR) as u32
    }
}

pub struct HnswIndex {
    dimension: usize,
    hnsw: RwLock<Hnsw<VectorPoint, Pcg64, 12, 24>>,
    // We group these to allow atomic swapping during reindex
    maps: RwLock<HnswMaps>,
    deleted_internal_ids: DashSet<usize>,
}

struct HnswMaps {
    id_to_ext: HashMap<usize, String>,
    ext_to_id: HashMap<String, usize>,
    internal_to_vector: HashMap<usize, Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
struct PersistedIndexData {
    vectors: Vec<(String, Vec<f32>)>,
}

impl HnswIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            hnsw: RwLock::new(Hnsw::new()),
            maps: RwLock::new(HnswMaps {
                id_to_ext: HashMap::new(),
                ext_to_id: HashMap::new(),
                internal_to_vector: HashMap::new(),
            }),
            deleted_internal_ids: DashSet::new(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let maps = self.maps.read().unwrap();
        let mut vectors = Vec::new();
        
        for (&internal_id, ext_id) in &maps.id_to_ext {
            if let Some(vec) = maps.internal_to_vector.get(&internal_id) {
                vectors.push((ext_id.clone(), vec.clone()));
            }
        }

        let persisted = PersistedIndexData { vectors };

        let file = File::create(path).map_err(|e| IndexError::Index(format!("Failed to create index file: {}", e)))?;
        bincode::serialize_into(file, &persisted)
            .map_err(|e| IndexError::Index(format!("Failed to serialize index: {}", e)))?;

        Ok(())
    }

    pub fn load(path: &Path, dimension: usize) -> Result<Self> {
        let file = File::open(path).map_err(|e| IndexError::Index(format!("Failed to open index file: {}", e)))?;
        let persisted: PersistedIndexData = bincode::deserialize_from(file)
            .map_err(|e| IndexError::Index(format!("Failed to deserialize index: {}", e)))?;

        let index = Self::new(dimension);
        for (id, vector) in persisted.vectors {
            let _ = index.add(id, &vector);
        }

        Ok(index)
    }

    pub fn count(&self) -> usize {
        self.maps.read().unwrap().ext_to_id.len()
    }

    pub fn reindex(&self) -> Result<()> {
        let mut hnsw_guard = self.hnsw.write().unwrap();
        let mut temp_hnsw = Hnsw::new();
        let mut searcher = Searcher::default();
        
        let mut new_id_to_ext = HashMap::new();
        let mut new_ext_to_id = HashMap::new();
        let mut new_internal_to_vector = HashMap::new();
        
        // Use a block to minimize maps read lock time
        let old_data = {
            let maps = self.maps.read().unwrap();
            maps.id_to_ext.iter()
                .filter(|(&internal_id, _)| !self.deleted_internal_ids.contains(&internal_id))
                .map(|(&internal_id, ext_id)| (ext_id.clone(), maps.internal_to_vector.get(&internal_id).cloned()))
                .collect::<Vec<_>>()
        };

        for (ext_id, vec_opt) in old_data {
            if let Some(vec) = vec_opt {
                let new_internal_id = temp_hnsw.insert(VectorPoint(vec.clone()), &mut searcher);
                new_id_to_ext.insert(new_internal_id, ext_id.clone());
                new_ext_to_id.insert(ext_id, new_internal_id);
                new_internal_to_vector.insert(new_internal_id, vec);
            }
        }

        // ATOMIC SWAP: Under hnsw_guard, swap both the graph and the maps
        *hnsw_guard = temp_hnsw;
        
        let mut maps_guard = self.maps.write().unwrap();
        maps_guard.id_to_ext = new_id_to_ext;
        maps_guard.ext_to_id = new_ext_to_id;
        maps_guard.internal_to_vector = new_internal_to_vector;
        
        self.deleted_internal_ids.clear();

        Ok(())
    }

    pub fn tombstone_count(&self) -> usize {
        self.deleted_internal_ids.len()
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
        
        let mut maps = self.maps.write().unwrap();
        maps.ext_to_id.insert(id.clone(), internal_id);
        maps.id_to_ext.insert(internal_id, id);
        maps.internal_to_vector.insert(internal_id, vector.to_vec());

        Ok(())
    }

    fn remove(&self, id: String) -> Result<()> {
        let mut maps = self.maps.write().unwrap();
        if let Some(internal_id) = maps.ext_to_id.remove(&id) {
            maps.id_to_ext.remove(&internal_id);
            self.deleted_internal_ids.insert(internal_id);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(IndexError::Index("Query dimension mismatch".to_string()));
        }

        let hnsw = self.hnsw.read().unwrap();
        let maps = self.maps.read().unwrap();
        let mut searcher = Searcher::default();
        
        let node_count = maps.id_to_ext.len() + self.deleted_internal_ids.len();
        if node_count == 0 {
            return Ok(Vec::new());
        }

        let mut ef = (k * 4).clamp(40, 250);
        let mut results = Vec::new();
        let mut attempts = 0;

        while results.len() < k && attempts < 3 {
            attempts += 1;
            
            // Safety guard for small collections to prevent hnsw crate panics
            let search_ef = if node_count < 2 {
                1
            } else if node_count < 100 {
                ef.min(node_count - 1).max(1)
            } else {
                ef.min(node_count - 1).max(1)
            };
            
            let mut neighbors = vec![Neighbor { index: 0, distance: 0 }; search_ef];
            let found_neighbors = hnsw.nearest(&VectorPoint(query.to_vec()), search_ef, &mut searcher, &mut neighbors);

            results.clear();
            for neighbor in found_neighbors {
                if self.deleted_internal_ids.contains(&neighbor.index) {
                    continue;
                }

                if let Some(ext_id) = maps.id_to_ext.get(&neighbor.index) {
                    let similarity = 1.0 - (neighbor.distance as f32 / SCALE_FACTOR);
                    results.push(SearchResult {
                        id: ext_id.clone(),
                        score: similarity,
                    });
                }
            }

            if results.len() >= k || ef >= 250 || ef >= node_count {
                break;
            }
            ef = (ef * 2).min(250);
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn count(&self) -> usize {
        self.uuid_to_id.len()
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
