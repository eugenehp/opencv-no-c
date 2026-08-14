//! Delaunay triangulation and Voronoi tessellation
//! (`modules/imgproc/src/subdivision2d.cpp`).
//!
//! # The quad-edge encoding
//!
//! Upstream stores the subdivision as Guibas-Stolfi quad-edges: each undirected
//! edge is four directed records — the edge, its dual, its reverse and the
//! reverse dual — packed so that an edge id's low two bits *are* the rotation.
//! `sym` is `id ^ 2`, `rot` is `(id & !3) + ((id + r) & 3)`, and
//! [`Subdiv2D::get_edge`] walks a neighbour by adding a two-nibble constant.
//!
//! That encoding is reproduced exactly rather than wrapped in something
//! friendlier, because the edge ids are part of the public API: upstream hands
//! them to callers, who pass them back to `getEdge` and `edgeOrg`. A different
//! numbering would give different ids for the same triangulation, and any code
//! that round-trips them through OpenCV would break.
//!
//! # Where this diverges
//!
//! The triangulation itself matches cv2: same vertex ids, same edge count, same
//! triangle set, and on simple inputs the same edge ids in the same order. On
//! larger point sets a handful of edge ids differ — the swap sequence that
//! restores the Delaunay condition can reach the same topology by a different
//! route, and the ids record the route. The geometry is identical; code that
//! stores an edge id from one library and reads it in the other is not
//! guaranteed to agree. That is a real limitation and it is stated here rather
//! than left to be discovered.

use cv_core::{cv_assert, Code, Error, Result};

/// Where a point fell, from [`Subdiv2D::locate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLocation {
    /// `cv::Subdiv2D::PTLOC_INSIDE` — inside a facet; the edge is one of its sides.
    Inside,
    /// `cv::Subdiv2D::PTLOC_ON_EDGE` — on an edge.
    OnEdge,
    /// `cv::Subdiv2D::PTLOC_VERTEX` — coincident with an existing vertex.
    Vertex,
    /// `cv::Subdiv2D::PTLOC_OUTSIDE_RECT` — outside the bounding rectangle.
    OutsideRect,
}

/// Which neighbour [`Subdiv2D::get_edge`] should return.
///
/// The values are upstream's, and they are not arbitrary: the low nibble picks
/// the rotation to read `next` at, the high nibble the rotation to apply to the
/// result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EdgeType {
    /// `cv::Subdiv2D::NEXT_AROUND_ORG`
    NextAroundOrg = 0x00,
    /// `cv::Subdiv2D::NEXT_AROUND_DST`
    NextAroundDst = 0x22,
    /// `cv::Subdiv2D::PREV_AROUND_ORG`
    PrevAroundOrg = 0x11,
    /// `cv::Subdiv2D::PREV_AROUND_DST`
    PrevAroundDst = 0x33,
    /// `cv::Subdiv2D::NEXT_AROUND_LEFT`
    NextAroundLeft = 0x13,
    /// `cv::Subdiv2D::NEXT_AROUND_RIGHT`
    NextAroundRight = 0x31,
    /// `cv::Subdiv2D::PREV_AROUND_LEFT`
    PrevAroundLeft = 0x20,
    /// `cv::Subdiv2D::PREV_AROUND_RIGHT`
    PrevAroundRight = 0x02,
}

#[derive(Debug, Clone, Copy)]
struct Vertex {
    pt: (f32, f32),
    is_virtual: bool,
    first_edge: i32,
}

#[derive(Debug, Clone, Copy)]
struct QuadEdge {
    next: [i32; 4],
    pt: [i32; 4],
}

impl QuadEdge {
    fn new(edgeidx: i32) -> Self {
        Self { next: [edgeidx, edgeidx + 3, edgeidx + 2, edgeidx + 1], pt: [0; 4] }
    }
    fn blank() -> Self {
        Self { next: [0; 4], pt: [0; 4] }
    }
}

/// One Voronoi cell: its corner polygon and the point it surrounds.
pub type VoronoiFacet = (Vec<(f32, f32)>, (f32, f32));

/// `cv::Subdiv2D` — an incremental Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct Subdiv2D {
    vtx: Vec<Vertex>,
    qedges: Vec<QuadEdge>,
    free_qedge: i32,
    free_point: i32,
    recent_edge: i32,
    top_left: (f32, f32),
    bottom_right: (f32, f32),
}

/// The signed area of the triangle `abc`, doubled.
fn triangle_area(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f64 {
    (f64::from(b.0) - f64::from(a.0)) * (f64::from(c.1) - f64::from(a.1))
        - (f64::from(b.1) - f64::from(a.1)) * (f64::from(c.0) - f64::from(a.0))
}

fn is_right_of(pt: (f32, f32), org: (f32, f32), dst: (f32, f32)) -> i32 {
    // Upstream is `triangleArea(pt, dst, org)` — destination before origin.
    // Writing the arguments the other way round negates it, which flips every
    // branch of the point-location walk and builds a plausible-looking but
    // wrong triangulation.
    let cw = triangle_area(pt, dst, org);
    i32::from(cw > 0.0) - i32::from(cw < 0.0)
}

impl Subdiv2D {
    /// `cv::Subdiv2D::initDelaunay(rect)`.
    ///
    /// The bindings expose this as `cv::Subdiv2D::initDelaunay2f(rect)`, the
    /// `Rect2f` overload — the arguments here are that rectangle's components.
    ///
    /// Seeds the subdivision with a virtual triangle far outside `rect`, so
    /// every real point inserted later lands strictly inside something. The
    /// virtual vertices are filtered back out by [`Self::get_triangle_list`].
    pub fn init_delaunay(x: f32, y: f32, width: f32, height: f32) -> Result<Self> {
        cv_assert!(width > 0.0 && height > 0.0, "the bounding rectangle must be non-empty");
        let big = 3.0 * width.max(height);
        let mut s = Self {
            // Index 0 of both arrays is a sentinel and never used, matching
            // upstream — edge id 0 means "no edge".
            vtx: vec![Vertex { pt: (0.0, 0.0), is_virtual: false, first_edge: 0 }],
            qedges: vec![QuadEdge::blank()],
            free_qedge: 0,
            free_point: 0,
            recent_edge: 0,
            top_left: (x, y),
            bottom_right: (x + width, y + height),
        };
        let a = s.new_point((x + big, y), false);
        let b = s.new_point((x, y + big), false);
        let c = s.new_point((x - big, y - big), false);
        let ea = s.new_edge();
        let eb = s.new_edge();
        let ec = s.new_edge();
        s.set_edge_points(ea, a, b);
        s.set_edge_points(eb, b, c);
        s.set_edge_points(ec, c, a);
        let sc = Self::sym_edge(ec);
        s.splice(ea, sc);
        let sa = Self::sym_edge(ea);
        s.splice(eb, sa);
        let sb = Self::sym_edge(eb);
        s.splice(ec, sb);
        s.recent_edge = ea;
        Ok(s)
    }

    fn new_edge(&mut self) -> i32 {
        if self.free_qedge <= 0 {
            self.qedges.push(QuadEdge::blank());
            self.free_qedge = self.qedges.len() as i32 - 1;
        }
        let edge = self.free_qedge * 4;
        self.free_qedge = self.qedges[(edge >> 2) as usize].next[1];
        self.qedges[(edge >> 2) as usize] = QuadEdge::new(edge);
        edge
    }

    fn delete_edge(&mut self, edge: i32) {
        let prev = self.get_edge(edge, EdgeType::PrevAroundOrg);
        self.splice(edge, prev);
        let s = Self::sym_edge(edge);
        let sprev = self.get_edge(s, EdgeType::PrevAroundOrg);
        self.splice(s, sprev);
        let q = (edge >> 2) as usize;
        self.qedges[q].next[1] = self.free_qedge;
        self.qedges[q].pt = [0; 4];
        self.free_qedge = q as i32;
    }

    fn new_point(&mut self, pt: (f32, f32), is_virtual: bool) -> i32 {
        if self.free_point == 0 {
            self.vtx.push(Vertex { pt: (0.0, 0.0), is_virtual: false, first_edge: 0 });
            self.free_point = self.vtx.len() as i32 - 1;
        }
        let vidx = self.free_point;
        self.free_point = self.vtx[vidx as usize].first_edge;
        self.vtx[vidx as usize] = Vertex { pt, is_virtual, first_edge: 0 };
        vidx
    }

    fn set_edge_points(&mut self, edge: i32, org: i32, dst: i32) {
        let q = (edge >> 2) as usize;
        self.qedges[q].pt[(edge & 3) as usize] = org;
        self.qedges[q].pt[((edge + 2) & 3) as usize] = dst;
        self.vtx[org as usize].first_edge = edge;
        self.vtx[dst as usize].first_edge = Self::sym_edge(edge);
    }

    fn splice(&mut self, a: i32, b: i32) {
        let (qa, ia) = ((a >> 2) as usize, (a & 3) as usize);
        let (qb, ib) = ((b >> 2) as usize, (b & 3) as usize);
        let a_next = self.qedges[qa].next[ia];
        let b_next = self.qedges[qb].next[ib];
        let a_rot = Self::rotate_edge(a_next, 1);
        let b_rot = Self::rotate_edge(b_next, 1);
        let (qar, iar) = ((a_rot >> 2) as usize, (a_rot & 3) as usize);
        let (qbr, ibr) = ((b_rot >> 2) as usize, (b_rot & 3) as usize);
        let t = self.qedges[qar].next[iar];
        self.qedges[qar].next[iar] = self.qedges[qbr].next[ibr];
        self.qedges[qbr].next[ibr] = t;
        // Upstream holds `a_next`/`b_next` as C++ *references* into this same
        // array, so the swap above can have just overwritten them. Re-reading
        // rather than reusing the values captured earlier is what makes this
        // match: with stale copies the ring closes wrongly and the whole
        // triangulation comes out subtly malformed.
        let a_next = self.qedges[qa].next[ia];
        let b_next = self.qedges[qb].next[ib];
        self.qedges[qa].next[ia] = b_next;
        self.qedges[qb].next[ib] = a_next;
    }

    fn connect_edges(&mut self, a: i32, b: i32) -> i32 {
        let edge = self.new_edge();
        let al = self.get_edge(a, EdgeType::NextAroundLeft);
        self.splice(edge, al);
        let se = Self::sym_edge(edge);
        self.splice(se, b);
        let (od, ob) = (self.edge_dst(a), self.edge_org(b));
        self.set_edge_points(edge, od, ob);
        edge
    }

    fn swap_edges(&mut self, edge: i32) {
        let sedge = Self::sym_edge(edge);
        let a = self.get_edge(edge, EdgeType::PrevAroundOrg);
        let b = self.get_edge(sedge, EdgeType::PrevAroundOrg);
        self.splice(edge, a);
        self.splice(sedge, b);
        let (da, db) = (self.edge_dst(a), self.edge_dst(b));
        self.set_edge_points(edge, da, db);
        let al = self.get_edge(a, EdgeType::NextAroundLeft);
        self.splice(edge, al);
        let bl = self.get_edge(b, EdgeType::NextAroundLeft);
        self.splice(sedge, bl);
    }

    /// `cv::Subdiv2D::rotateEdge(edge, rotate)`.
    #[must_use]
    pub fn rotate_edge(edge: i32, rotate: i32) -> i32 {
        (edge & !3) + ((edge + rotate) & 3)
    }

    /// `cv::Subdiv2D::symEdge(edge)` — the same edge, reversed.
    #[must_use]
    pub fn sym_edge(edge: i32) -> i32 {
        edge ^ 2
    }

    /// `cv::Subdiv2D::getEdge(edge, nextEdgeType)`.
    #[must_use]
    pub fn get_edge(&self, edge: i32, t: EdgeType) -> i32 {
        let t = t as i32;
        let e = self.qedges[(edge >> 2) as usize].next[((edge + t) & 3) as usize];
        (e & !3) + ((e + (t >> 4)) & 3)
    }

    /// `cv::Subdiv2D::nextEdge(edge)` — the next edge around the origin.
    #[must_use]
    pub fn next_edge(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].next[(edge & 3) as usize]
    }

    /// `cv::Subdiv2D::edgeOrg(edge)` — the origin vertex id and its position.
    #[must_use]
    pub fn edge_org_pt(&self, edge: i32) -> (i32, (f32, f32)) {
        let v = self.edge_org(edge);
        (v, self.vtx[v as usize].pt)
    }

    fn edge_org(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].pt[(edge & 3) as usize]
    }

    /// `cv::Subdiv2D::edgeDst(edge)` — the destination vertex id and position.
    #[must_use]
    pub fn edge_dst_pt(&self, edge: i32) -> (i32, (f32, f32)) {
        let v = self.edge_dst(edge);
        (v, self.vtx[v as usize].pt)
    }

    fn edge_dst(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].pt[((edge + 2) & 3) as usize]
    }

    /// `cv::Subdiv2D::getVertex(vertex)` — a vertex position and one incident edge.
    pub fn get_vertex(&self, vertex: i32) -> Result<((f32, f32), i32)> {
        let v = self
            .vtx
            .get(vertex as usize)
            .ok_or_else(|| Error::new(Code::StsOutOfRange, format!("no vertex {vertex}")))?;
        Ok((v.pt, v.first_edge))
    }

    fn is_right_of_edge(&self, pt: (f32, f32), edge: i32) -> i32 {
        is_right_of(pt, self.vtx[self.edge_org(edge) as usize].pt, self.vtx[self.edge_dst(edge) as usize].pt)
    }

    /// `cv::Subdiv2D::locate(pt, edge, vertex)`.
    ///
    /// Returns where `pt` sits, plus the edge and vertex upstream reports
    /// alongside it — the vertex is meaningful only for
    /// [`PointLocation::Vertex`], the edge only for the other two inside cases.
    pub fn locate(&mut self, pt: (f32, f32)) -> Result<(PointLocation, i32, i32)> {
        cv_assert!(self.qedges.len() >= 4, "the subdivision is empty; call init_delaunay first");
        if pt.0 < self.top_left.0
            || pt.1 < self.top_left.1
            || pt.0 >= self.bottom_right.0
            || pt.1 >= self.bottom_right.1
        {
            return Ok((PointLocation::OutsideRect, 0, 0));
        }

        let max_edges = self.qedges.len() as i32 * 4;
        let mut edge = self.recent_edge;
        cv_assert!(edge > 0, "the subdivision has no edges");
        let mut right_of_curr = self.is_right_of_edge(pt, edge);
        if right_of_curr > 0 {
            edge = Self::sym_edge(edge);
            right_of_curr = -right_of_curr;
        }
        let mut location = None;

        for _ in 0..max_edges {
            let onext = self.next_edge(edge);
            let dprev = self.get_edge(edge, EdgeType::PrevAroundDst);
            let right_of_onext = self.is_right_of_edge(pt, onext);
            let right_of_dprev = self.is_right_of_edge(pt, dprev);

            if right_of_dprev > 0 {
                if right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) {
                    location = Some(PointLocation::Inside);
                    break;
                }
                right_of_curr = right_of_onext;
                edge = onext;
            } else if right_of_onext > 0 {
                if right_of_dprev == 0 && right_of_curr == 0 {
                    location = Some(PointLocation::Inside);
                    break;
                }
                right_of_curr = right_of_dprev;
                edge = dprev;
            } else if right_of_curr == 0
                && self.is_right_of_edge(self.vtx[self.edge_dst(onext) as usize].pt, edge) >= 0
            {
                edge = Self::sym_edge(edge);
            } else {
                right_of_curr = right_of_onext;
                edge = onext;
            }
        }
        self.recent_edge = edge;

        let Some(mut loc) = location else {
            return Err(Error::new(
                Code::StsNoConv,
                "locate failed to settle; the subdivision may be corrupt".to_string(),
            ));
        };
        let mut vertex = 0;
        let mut out_edge = edge;

        let org = self.vtx[self.edge_org(edge) as usize].pt;
        let dst = self.vtx[self.edge_dst(edge) as usize].pt;
        let t1 = (f64::from(pt.0) - f64::from(org.0)).abs() + (f64::from(pt.1) - f64::from(org.1)).abs();
        let t2 = (f64::from(pt.0) - f64::from(dst.0)).abs() + (f64::from(pt.1) - f64::from(dst.1)).abs();
        let t3 = (f64::from(org.0) - f64::from(dst.0)).abs() + (f64::from(org.1) - f64::from(dst.1)).abs();
        let eps = f64::from(f32::EPSILON);

        if t1 < eps {
            loc = PointLocation::Vertex;
            vertex = self.edge_org(edge);
            out_edge = 0;
        } else if t2 < eps {
            loc = PointLocation::Vertex;
            vertex = self.edge_dst(edge);
            out_edge = 0;
        } else if (t1 < t3 || t2 < t3)
            && (triangle_area(pt, org, dst)).abs() * 2.0 < t3 * eps * t3
        {
            loc = PointLocation::OnEdge;
            vertex = 0;
        }
        Ok((loc, out_edge, vertex))
    }

    /// Upstream's in-circle predicate: is `pt` inside the circle through `a`, `b`, `c`?
    fn is_pt_in_circle3(pt: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> i32 {
        let eps = f64::from(f32::EPSILON) * 0.125;
        let sq = |p: (f32, f32)| f64::from(p.0) * f64::from(p.0) + f64::from(p.1) * f64::from(p.1);
        let mut val = sq(a) * triangle_area(b, c, pt);
        val -= sq(b) * triangle_area(a, c, pt);
        val += sq(c) * triangle_area(a, b, pt);
        val -= sq(pt) * triangle_area(a, b, c);
        if val > eps {
            1
        } else if val < -eps {
            -1
        } else {
            0
        }
    }

    /// `cv::Subdiv2D::insert(pt)` — add a point, returning its vertex id.
    ///
    /// Inserting a point that is already present returns the existing id
    /// rather than duplicating it, which is what upstream does.
    pub fn insert(&mut self, pt: (f32, f32)) -> Result<i32> {
        let (location, mut curr_edge, curr_vertex) = self.locate(pt)?;
        match location {
            PointLocation::OutsideRect => {
                return Err(Error::new(
                    Code::StsOutOfRange,
                    format!("({}, {}) is outside the subdivision's rectangle", pt.0, pt.1),
                ))
            }
            PointLocation::Vertex => return Ok(curr_vertex),
            PointLocation::OnEdge => {
                let deleted = curr_edge;
                curr_edge = self.get_edge(curr_edge, EdgeType::PrevAroundOrg);
                self.recent_edge = curr_edge;
                self.delete_edge(deleted);
            }
            PointLocation::Inside => {}
        }
        cv_assert!(curr_edge != 0, "locate returned no edge to attach to");

        let curr_point = self.new_point(pt, false);
        let mut base_edge = self.new_edge();
        let first_point = self.edge_org(curr_edge);
        self.set_edge_points(base_edge, first_point, curr_point);
        self.splice(base_edge, curr_edge);

        // Fan out to every vertex of the containing facet.
        loop {
            let s = Self::sym_edge(base_edge);
            base_edge = self.connect_edges(curr_edge, s);
            curr_edge = self.get_edge(base_edge, EdgeType::PrevAroundOrg);
            if self.edge_dst(curr_edge) == first_point {
                break;
            }
        }
        curr_edge = self.get_edge(base_edge, EdgeType::PrevAroundOrg);

        // Restore the Delaunay condition by flipping edges that fail it.
        let max_edges = self.qedges.len() as i32 * 4;
        for _ in 0..max_edges {
            let temp_edge = self.get_edge(curr_edge, EdgeType::PrevAroundOrg);
            let temp_dst = self.edge_dst(temp_edge);
            let curr_org = self.edge_org(curr_edge);
            let curr_dst = self.edge_dst(curr_edge);

            if self.is_right_of_edge(self.vtx[temp_dst as usize].pt, curr_edge) > 0
                && Self::is_pt_in_circle3(
                    self.vtx[curr_org as usize].pt,
                    self.vtx[temp_dst as usize].pt,
                    self.vtx[curr_dst as usize].pt,
                    self.vtx[curr_point as usize].pt,
                ) < 0
            {
                self.swap_edges(curr_edge);
                curr_edge = self.get_edge(curr_edge, EdgeType::PrevAroundOrg);
            } else if curr_org == first_point {
                break;
            } else {
                let n = self.next_edge(curr_edge);
                curr_edge = self.get_edge(n, EdgeType::PrevAroundLeft);
            }
        }
        Ok(curr_point)
    }

    /// `cv::Subdiv2D::insert(ptvec)` — insert many points.
    pub fn insert_all(&mut self, pts: &[(f32, f32)]) -> Result<Vec<i32>> {
        pts.iter().map(|p| self.insert(*p)).collect()
    }

    /// `cv::Subdiv2D::findNearest(pt, nearestPt)` — the closest inserted vertex.
    pub fn find_nearest(&mut self, pt: (f32, f32)) -> Result<Option<(i32, (f32, f32))>> {
        let (location, edge, vertex) = self.locate(pt)?;
        if location == PointLocation::OutsideRect {
            return Ok(None);
        }
        if location == PointLocation::Vertex {
            return Ok(Some((vertex, self.vtx[vertex as usize].pt)));
        }
        // Walk the facet's ring, keeping the closest real vertex seen.
        let mut best: Option<(i32, f64)> = None;
        let start = if edge != 0 { edge } else { self.recent_edge };
        let mut e = start;
        for _ in 0..(self.qedges.len() * 4) {
            for v in [self.edge_org(e), self.edge_dst(e)] {
                if v > 0 && !self.vtx[v as usize].is_virtual {
                    let q = self.vtx[v as usize].pt;
                    let d = (f64::from(q.0) - f64::from(pt.0)).powi(2)
                        + (f64::from(q.1) - f64::from(pt.1)).powi(2);
                    if best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((v, d));
                    }
                }
            }
            e = self.get_edge(e, EdgeType::NextAroundLeft);
            if e == start {
                break;
            }
        }
        Ok(best.map(|(v, _)| (v, self.vtx[v as usize].pt)))
    }

    /// `cv::Subdiv2D::getEdgeList(edgeList)` — every edge as `(x0, y0, x1, y1)`.
    ///
    /// Starts at quad-edge 4, skipping the seed triangle, exactly as upstream does.
    #[must_use]
    pub fn get_edge_list(&self) -> Vec<[f32; 4]> {
        let mut out = Vec::new();
        for q in self.qedges.iter().skip(4) {
            if q.pt[0] <= 0 || q.pt[2] <= 0 {
                continue;
            }
            let a = self.vtx[q.pt[0] as usize].pt;
            let b = self.vtx[q.pt[2] as usize].pt;
            out.push([a.0, a.1, b.0, b.1]);
        }
        out
    }

    /// `cv::Subdiv2D::getLeadingEdgeList(leadingEdgeList)` — one edge per triangle.
    #[must_use]
    pub fn get_leading_edge_list(&self) -> Vec<i32> {
        let total = self.qedges.len() * 4;
        let mut mask = vec![false; total];
        let mut out = Vec::new();
        for i in (4..total).step_by(2) {
            if mask[i] {
                continue;
            }
            let mut e = i as i32;
            mask[e as usize] = true;
            e = self.get_edge(e, EdgeType::NextAroundLeft);
            mask[e as usize] = true;
            e = self.get_edge(e, EdgeType::NextAroundLeft);
            mask[e as usize] = true;
            out.push(i as i32);
        }
        out
    }

    /// `cv::Subdiv2D::getTriangleList(triangleList)`.
    ///
    /// Triangles with a corner outside the bounding rectangle are dropped —
    /// those touch the seed triangle's virtual vertices and are an artefact of
    /// the construction, not part of the data's triangulation.
    ///
    /// A rejected triangle leaves the edge mask untouched, which matters: mark
    /// its edges and a legitimate triangle sharing one of them is silently lost
    /// later in the sweep.
    #[must_use]
    pub fn get_triangle_list(&self) -> Vec<[f32; 6]> {
        let total = self.qedges.len() * 4;
        let mut mask = vec![false; total];
        let mut out = Vec::new();
        for i in (4..total).step_by(2) {
            if mask[i] {
                continue;
            }
            let ea = i as i32;
            let a = self.vtx[self.edge_org(ea).max(0) as usize].pt;
            if !self.in_rect(a) {
                continue;
            }
            let eb = self.get_edge(ea, EdgeType::NextAroundLeft);
            let b = self.vtx[self.edge_org(eb).max(0) as usize].pt;
            if !self.in_rect(b) {
                continue;
            }
            let ec = self.get_edge(eb, EdgeType::NextAroundLeft);
            let c = self.vtx[self.edge_org(ec).max(0) as usize].pt;
            if !self.in_rect(c) {
                continue;
            }
            mask[ea as usize] = true;
            mask[eb as usize] = true;
            mask[ec as usize] = true;
            out.push([a.0, a.1, b.0, b.1, c.0, c.1]);
        }
        out
    }

    /// Upstream's `Rect2f::contains`: low edges inclusive, high edges exclusive.
    fn in_rect(&self, p: (f32, f32)) -> bool {
        p.0 >= self.top_left.0
            && p.1 >= self.top_left.1
            && p.0 < self.bottom_right.0
            && p.1 < self.bottom_right.1
    }

    /// `cv::Subdiv2D::getVoronoiFacetList(idx, facetList, facetCenters)`.
    ///
    /// The Voronoi diagram is the Delaunay triangulation's dual: each facet is
    /// the region closer to its centre than to any other inserted point, and
    /// its corners are the circumcentres of the triangles around that point.
    ///
    /// Pass an empty `idx` for every vertex.
    pub fn get_voronoi_facet_list(&self, idx: &[i32]) -> Result<Vec<VoronoiFacet>> {
        let ids: Vec<i32> = if idx.is_empty() {
            (1..self.vtx.len() as i32)
                .filter(|&v| !self.vtx[v as usize].is_virtual && self.vtx[v as usize].first_edge != 0)
                .filter(|&v| self.in_rect(self.vtx[v as usize].pt))
                .collect()
        } else {
            idx.to_vec()
        };
        let mut out = Vec::new();
        for v in ids {
            let vtx = *self
                .vtx
                .get(v as usize)
                .ok_or_else(|| Error::new(Code::StsOutOfRange, format!("no vertex {v}")))?;
            if vtx.first_edge == 0 {
                continue;
            }
            let mut facet = Vec::new();
            let start = vtx.first_edge;
            let mut e = start;
            for _ in 0..(self.qedges.len() * 4) {
                // The circumcentre of the triangle to this edge's left.
                let a = self.vtx[self.edge_org(e) as usize].pt;
                let b = self.vtx[self.edge_dst(e) as usize].pt;
                let nxt = self.get_edge(e, EdgeType::NextAroundLeft);
                let c = self.vtx[self.edge_dst(nxt) as usize].pt;
                if let Some(cc) = circumcenter(a, b, c) {
                    facet.push(cc);
                }
                e = self.next_edge(e);
                if e == start {
                    break;
                }
            }
            if !facet.is_empty() {
                out.push((facet, vtx.pt));
            }
        }
        Ok(out)
    }
}

/// The circumcentre of a triangle, or `None` when the points are collinear and
/// the circle has no finite centre.
fn circumcenter(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> Option<(f32, f32)> {
    let d = 2.0 * triangle_area(a, b, c);
    if d.abs() < 1e-12 {
        return None;
    }
    let (ax, ay) = (f64::from(a.0), f64::from(a.1));
    let (bx, by) = (f64::from(b.0), f64::from(b.1));
    let (cx, cy) = (f64::from(c.0), f64::from(c.1));
    let sa = ax * ax + ay * ay;
    let sb = bx * bx + by * by;
    let sc = cx * cx + cy * cy;
    let ux = (sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)) / d;
    let uy = (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / d;
    Some((ux as f32, uy as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_subdiv() -> Subdiv2D {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 100.0, 100.0).unwrap();
        for p in [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)] {
            s.insert(p).unwrap();
        }
        s
    }

    #[test]
    fn the_edge_encoding_matches_upstreams() {
        // sym is a two-bit rotation, so applying it twice is the identity.
        for e in [4, 5, 6, 7, 12, 33] {
            assert_eq!(Subdiv2D::sym_edge(Subdiv2D::sym_edge(e)), e);
            assert_eq!(Subdiv2D::sym_edge(e), Subdiv2D::rotate_edge(e, 2));
            // Four rotations return to the start.
            assert_eq!(Subdiv2D::rotate_edge(Subdiv2D::rotate_edge(e, 1), 3), e);
            // The quad-edge index is preserved; only the low two bits move.
            assert_eq!(Subdiv2D::rotate_edge(e, 1) >> 2, e >> 2);
        }
        // The EdgeType constants carry upstream's exact values, because callers
        // pass them across the API boundary.
        assert_eq!(EdgeType::NextAroundOrg as i32, 0x00);
        assert_eq!(EdgeType::NextAroundDst as i32, 0x22);
        assert_eq!(EdgeType::PrevAroundOrg as i32, 0x11);
        assert_eq!(EdgeType::PrevAroundDst as i32, 0x33);
        assert_eq!(EdgeType::NextAroundLeft as i32, 0x13);
        assert_eq!(EdgeType::NextAroundRight as i32, 0x31);
        assert_eq!(EdgeType::PrevAroundLeft as i32, 0x20);
        assert_eq!(EdgeType::PrevAroundRight as i32, 0x02);
    }

    #[test]
    fn init_delaunay_validates_its_rectangle() {
        assert!(Subdiv2D::init_delaunay(0.0, 0.0, 0.0, 10.0).is_err());
        assert!(Subdiv2D::init_delaunay(0.0, 0.0, 10.0, -1.0).is_err());
        assert!(Subdiv2D::init_delaunay(0.0, 0.0, 10.0, 10.0).is_ok());
    }

    #[test]
    fn inserting_a_point_returns_a_stable_id() {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 100.0, 100.0).unwrap();
        let a = s.insert((30.0, 40.0)).unwrap();
        // The first real vertex comes after the three virtual ones.
        assert_eq!(a, 4, "vertices 1..3 are the seed triangle");
        let b = s.insert((60.0, 70.0)).unwrap();
        assert_eq!(b, 5);
        // Re-inserting an existing point returns its id rather than duplicating.
        assert_eq!(s.insert((30.0, 40.0)).unwrap(), a);
        assert_eq!(s.insert((60.0, 70.0)).unwrap(), b);
    }

    #[test]
    fn a_point_outside_the_rectangle_is_refused() {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(s.insert((-5.0, 50.0)).is_err());
        assert!(s.insert((150.0, 50.0)).is_err());
        assert!(s.insert((50.0, 100.0)).is_err(), "the far edge is exclusive");
        // And locate says so rather than erroring.
        assert_eq!(s.locate((-5.0, 50.0)).unwrap().0, PointLocation::OutsideRect);
    }

    #[test]
    fn locate_finds_vertices_edges_and_interiors() {
        let mut s = square_subdiv();
        // Exactly on an inserted point.
        let (loc, _, v) = s.locate((20.0, 20.0)).unwrap();
        assert_eq!(loc, PointLocation::Vertex);
        assert_eq!(s.get_vertex(v).unwrap().0, (20.0, 20.0));
        // Well inside a facet.
        let (loc, e, _) = s.locate((50.0, 30.0)).unwrap();
        assert_eq!(loc, PointLocation::Inside);
        assert_ne!(e, 0, "an interior hit names the facet's edge");
        // The midpoint of the top side lies on the edge joining its ends.
        let (loc, _, _) = s.locate((50.0, 20.0)).unwrap();
        assert!(
            matches!(loc, PointLocation::OnEdge | PointLocation::Inside),
            "a point on a side is on an edge or inside, got {loc:?}"
        );
    }

    #[test]
    fn a_square_triangulates_into_two_triangles() {
        let s = square_subdiv();
        let tris = s.get_triangle_list();
        assert_eq!(tris.len(), 2, "four corners make two triangles, got {tris:?}");
        // Every vertex of every triangle is one of the four inserted points.
        let corners = [(20.0f32, 20.0f32), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)];
        for t in &tris {
            for k in 0..3 {
                let p = (t[k * 2], t[k * 2 + 1]);
                assert!(corners.contains(&p), "unexpected vertex {p:?}");
            }
            // And the triangle is non-degenerate.
            let a = (t[0], t[1]);
            let b = (t[2], t[3]);
            let c = (t[4], t[5]);
            assert!(triangle_area(a, b, c).abs() > 1.0, "degenerate triangle {t:?}");
        }
        // Together they cover the square's area.
        let total: f64 = tris
            .iter()
            .map(|t| triangle_area((t[0], t[1]), (t[2], t[3]), (t[4], t[5])).abs() / 2.0)
            .sum();
        assert!((total - 3600.0).abs() < 1.0, "area {total} should be 60x60");
    }

    #[test]
    fn the_triangulation_satisfies_the_delaunay_condition() {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 200.0, 200.0).unwrap();
        let pts: Vec<(f32, f32)> = (0..24)
            .map(|i| {
                let f = f64::from(i);
                (
                    (20.0 + 80.0 * (1.0 + (f * 1.7).sin())) as f32,
                    (20.0 + 80.0 * (1.0 + (f * 2.3).cos())) as f32,
                )
            })
            .collect();
        s.insert_all(&pts).unwrap();
        let tris = s.get_triangle_list();
        assert!(tris.len() >= 10, "expected a real triangulation, got {}", tris.len());

        // No inserted point may lie strictly inside any triangle's circumcircle.
        for t in &tris {
            let (a, b, c) = ((t[0], t[1]), (t[2], t[3]), (t[4], t[5]));
            let Some(cc) = circumcenter(a, b, c) else { continue };
            let r2 = (f64::from(cc.0) - f64::from(a.0)).powi(2)
                + (f64::from(cc.1) - f64::from(a.1)).powi(2);
            for p in &pts {
                if [a, b, c].contains(p) {
                    continue;
                }
                let d2 = (f64::from(cc.0) - f64::from(p.0)).powi(2)
                    + (f64::from(cc.1) - f64::from(p.1)).powi(2);
                assert!(
                    d2 >= r2 - 1e-3 * r2.max(1.0),
                    "{p:?} lies inside the circumcircle of {t:?} — not Delaunay"
                );
            }
        }
    }

    #[test]
    fn every_edge_is_reported_once_with_both_endpoints() {
        let s = square_subdiv();
        let edges = s.get_edge_list();
        assert!(!edges.is_empty());
        for e in &edges {
            assert!(
                (e[0] - e[2]).abs() > 0.0 || (e[1] - e[3]).abs() > 0.0,
                "an edge must have two distinct ends: {e:?}"
            );
        }
        // The two triangles share one diagonal, so five edges connect the four
        // corners; the rest run out to the virtual seed triangle.
        let real = |p: (f32, f32)| (20.0..=80.0).contains(&p.0) && (20.0..=80.0).contains(&p.1);
        let inner = edges.iter().filter(|e| real((e[0], e[1])) && real((e[2], e[3]))).count();
        assert_eq!(inner, 5, "four sides and one diagonal");
    }

    #[test]
    fn find_nearest_returns_the_closest_inserted_point() {
        let mut s = square_subdiv();
        for (query, want) in [
            ((25.0f32, 25.0f32), (20.0f32, 20.0f32)),
            ((75.0, 25.0), (80.0, 20.0)),
            ((78.0, 78.0), (80.0, 80.0)),
            ((22.0, 75.0), (20.0, 80.0)),
        ] {
            let (_, pt) = s.find_nearest(query).unwrap().expect("a nearest point exists");
            assert_eq!(pt, want, "nearest to {query:?}");
        }
        // Outside the rectangle there is nothing to return.
        assert!(s.find_nearest((-10.0, -10.0)).unwrap().is_none());
        // And an exact hit returns itself.
        assert_eq!(s.find_nearest((80.0, 20.0)).unwrap().unwrap().1, (80.0, 20.0));
    }

    #[test]
    fn voronoi_facets_are_dual_to_the_triangulation() {
        let mut s = square_subdiv();
        s.insert((50.0, 50.0)).unwrap();
        let facets = s.get_voronoi_facet_list(&[]).unwrap();
        assert_eq!(facets.len(), 5, "one facet per inserted point");
        for (poly, centre) in &facets {
            assert!(poly.len() >= 3, "a facet is a polygon, got {} corners", poly.len());
            // Every corner is equidistant-ish from its centre by construction;
            // the real invariant is that the centre is one of the inputs.
            assert!(
                [(20.0f32, 20.0f32), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0), (50.0, 50.0)]
                    .contains(centre),
                "unexpected facet centre {centre:?}"
            );
        }
        // The centre point's facet is bounded, and its corners surround it.
        let (poly, _) = facets.iter().find(|(_, c)| *c == (50.0, 50.0)).unwrap();
        let xs: Vec<f32> = poly.iter().map(|p| p.0).collect();
        let ys: Vec<f32> = poly.iter().map(|p| p.1).collect();
        assert!(xs.iter().any(|&x| x < 50.0) && xs.iter().any(|&x| x > 50.0));
        assert!(ys.iter().any(|&y| y < 50.0) && ys.iter().any(|&y| y > 50.0));
    }

    #[test]
    fn get_vertex_reports_position_and_an_incident_edge() {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 100.0, 100.0).unwrap();
        let v = s.insert((33.0, 44.0)).unwrap();
        let (pt, edge) = s.get_vertex(v).unwrap();
        assert_eq!(pt, (33.0, 44.0));
        assert_ne!(edge, 0, "an inserted vertex has at least one edge");
        // That edge really does start there.
        let (org, org_pt) = s.edge_org_pt(edge);
        assert_eq!(org, v);
        assert_eq!(org_pt, (33.0, 44.0));
        // And walking to its far end lands somewhere else.
        let (dst, _) = s.edge_dst_pt(edge);
        assert_ne!(dst, v);
        assert!(s.get_vertex(9999).is_err());
    }

    #[test]
    fn edge_navigation_round_trips() {
        let s = square_subdiv();
        let edges = s.get_leading_edge_list();
        assert!(!edges.is_empty(), "there is at least one triangle");
        for &e in &edges {
            // Three next-around-left steps close a triangle.
            let mut w = e;
            for _ in 0..3 {
                w = s.get_edge(w, EdgeType::NextAroundLeft);
            }
            assert_eq!(w, e, "next-around-left must close after three steps");
            // Org and dst swap under sym.
            let (o, _) = s.edge_org_pt(e);
            let (d, _) = s.edge_dst_pt(e);
            let se = Subdiv2D::sym_edge(e);
            assert_eq!(s.edge_org_pt(se).0, d);
            assert_eq!(s.edge_dst_pt(se).0, o);
            // Next and prev around the origin are inverses.
            let n = s.get_edge(e, EdgeType::NextAroundOrg);
            assert_eq!(s.get_edge(n, EdgeType::PrevAroundOrg), e);
        }
    }

    /// The same subdivision built in cv2 4.13.0.
    ///
    /// Vertex ids, edge count and the triangle set all match cv2 exactly. The
    /// triangles are compared as a set because cv2 emits them in edge-id order
    /// and each starting at whichever corner its scan reached first — see
    /// `matches_cv2_edge_ids_on_a_simple_case` for where the ids do line up,
    /// and the module docs for where they do not.
    #[test]
    fn matches_cv2_geometrically() {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 200.0, 200.0).unwrap();
        let pts: [(f32, f32); 8] = [
            (30.0, 40.0), (150.0, 50.0), (120.0, 170.0), (40.0, 160.0),
            (90.0, 100.0), (60.0, 70.0), (170.0, 120.0), (25.0, 110.0),
        ];
        assert_eq!(s.insert_all(&pts).unwrap(), vec![4, 5, 6, 7, 8, 9, 10, 11], "vertex ids");
        assert_eq!(s.get_edge_list().len(), 24, "edge count");
        assert_eq!(s.get_leading_edge_list().len(), 18, "one leading edge per facet");

        // cv2's getTriangleList, canonicalised: each triangle's corners sorted,
        // then the triangles sorted, so neither rotation nor emission order
        // affects the comparison.
        let canon = |v: Vec<[f32; 6]>| {
            let mut t: Vec<[(u32, u32); 3]> = v
                .iter()
                .map(|r| {
                    let mut c = [
                        (r[0] as u32, r[1] as u32),
                        (r[2] as u32, r[3] as u32),
                        (r[4] as u32, r[5] as u32),
                    ];
                    c.sort_unstable();
                    c
                })
                .collect();
            t.sort_unstable();
            t
        };
        let cv2: Vec<[f32; 6]> = vec![
            [30.0, 40.0, 150.0, 50.0, 60.0, 70.0],
            [170.0, 120.0, 90.0, 100.0, 150.0, 50.0],
            [90.0, 100.0, 170.0, 120.0, 120.0, 170.0],
            [90.0, 100.0, 120.0, 170.0, 40.0, 160.0],
            [60.0, 70.0, 25.0, 110.0, 30.0, 40.0],
            [25.0, 110.0, 60.0, 70.0, 90.0, 100.0],
            [150.0, 50.0, 90.0, 100.0, 60.0, 70.0],
            [90.0, 100.0, 40.0, 160.0, 25.0, 110.0],
        ];
        assert_eq!(canon(s.get_triangle_list()), canon(cv2), "triangle set");

        // locate and findNearest agree on the answers that are geometric.
        assert_eq!(s.locate((30.0, 40.0)).unwrap(), (PointLocation::Vertex, 0, 4));
        assert_eq!(s.locate((100.0, 100.0)).unwrap().0, PointLocation::Inside);
        assert_eq!(s.find_nearest((35.0, 45.0)).unwrap().unwrap().1, (30.0, 40.0));
        assert_eq!(s.find_nearest((160.0, 60.0)).unwrap().unwrap().1, (150.0, 50.0));
    }

    /// Where the edge ids do match cv2, they are pinned — they are API surface.
    #[test]
    fn matches_cv2_edge_ids_on_a_simple_case() {
        let mut s = Subdiv2D::init_delaunay(0.0, 0.0, 100.0, 100.0).unwrap();
        let ids = s
            .insert_all(&[(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)])
            .unwrap();
        assert_eq!(ids, vec![4, 5, 6, 7]);
        // cv2 4.13.0, byte for byte.
        assert_eq!(
            s.get_triangle_list(),
            vec![
                [20.0, 20.0, 80.0, 80.0, 20.0, 80.0],
                [80.0, 80.0, 20.0, 20.0, 80.0, 20.0],
            ]
        );
        assert_eq!(s.get_leading_edge_list(), vec![4, 6, 8, 12, 18, 22, 30, 32, 34, 46]);
        assert_eq!(s.get_edge_list().len(), 12);
        assert_eq!(s.locate((20.0, 20.0)).unwrap(), (PointLocation::Vertex, 0, 4));
    }
}
