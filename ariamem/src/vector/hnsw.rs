use crate::vector::index::{IndexError, Result, SearchResult, VectorIndex};
use dashmap::DashMap;
use hnsw::{Hnsw, Searcher};
use space::MetricPoint;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use uuid::Uuid;
use rand_pcg::Pcg64;

#[derive(Clone)]
pub struct VectorPoint(pub Vec<f32>);

impl MetricPoint for VectorPoint {
    type Metric = u32;

    fn distance(&self, other: &Self) -> Self::Metric {
        let a = &self.0;
        let b = &other.0;
        
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        let similarity = if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        };

        // Convert similarity [-1.0, 1.0] to distance [0.0, 2.0]
        let distance = 1.0 - similarity;
        
        // Scale to u32 for the space crate requirements
        (distance * 100_000.0) as u32
    }
}

pub struct HnswIndex {
    dimension: usize,
    hnsw: RwLock<Hnsw<VectorPoint, Pcg64, 16, 32>>,
    searcher: RwLock<Searcher<u32>>,
    id_to_uuid: DashMap<usize, Uuid>,
    uuid_to_id: DashMap<Uuid, usize>,
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
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(IndexError::Index("Query dimension mismatch".to_string()));
        }

        let hnsw = self.hnsw.read().unwrap();
        let mut searcher = self.searcher.write().unwrap();
        
        let mut neighbors = vec![];
        hnsw.nearest(&VectorPoint(query.to_vec()), 24, &mut searcher, &mut neighbors);

        let mut results = Vec::new();
        for neighbor in neighbors.iter().take(k) {
            if let Some(uuid) = self.id_to_uuid.get(&neighbor.index) {
                let similarity = 1.0 - (neighbor.distance as f32 / 100_000.0);
                results.push(SearchResult {
                    id: *uuid.value(),
                    score: similarity,
                });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}