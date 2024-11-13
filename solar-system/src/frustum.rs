use nalgebra_glm::{cross, dot, normalize, Vec3, Mat4};

pub struct Plane {
    normal: Vec3,
    d: f32,
}

impl Plane {
    pub fn from_points(p1: &Vec3, p2: &Vec3, p3: &Vec3) -> Self {
        let u = p2 - p1;
        let v = p3 - p1;
        let normal = normalize(&cross(&u, &v));
        let d = -dot(&normal, p1);
        Plane { normal, d }
    }

    pub fn normalize(&mut self) {
        let length = self.normal.magnitude();
        self.normal /= length;
        self.d /= length;
    }

    pub fn distance_to_point(&self, point: &Vec3) -> f32 {
        dot(&self.normal, point) + self.d
    }
}

pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    pub fn from_matrix(matrix: &Mat4) -> Self {
        let planes = [
            // Left
            Plane {
                normal: Vec3::new(matrix[3] + matrix[0], matrix[7] + matrix[4], matrix[11] + matrix[8]),
                d: matrix[15] + matrix[12],
            },
            // Right
            Plane {
                normal: Vec3::new(matrix[3] - matrix[0], matrix[7] - matrix[4], matrix[11] - matrix[8]),
                d: matrix[15] - matrix[12],
            },
            // Bottom
            Plane {
                normal: Vec3::new(matrix[3] + matrix[1], matrix[7] + matrix[5], matrix[11] + matrix[9]),
                d: matrix[15] + matrix[13],
            },
            // Top
            Plane {
                normal: Vec3::new(matrix[3] - matrix[1], matrix[7] - matrix[5], matrix[11] - matrix[9]),
                d: matrix[15] - matrix[13],
            },
            // Near
            Plane {
                normal: Vec3::new(matrix[3] + matrix[2], matrix[7] + matrix[6], matrix[11] + matrix[10]),
                d: matrix[15] + matrix[14],
            },
            // Far
            Plane {
                normal: Vec3::new(matrix[3] - matrix[2], matrix[7] - matrix[6], matrix[11] - matrix[10]),
                d: matrix[15] - matrix[14],
            },
        ];

        // Normalize the planes
        let mut frustum = Frustum { planes };
        for plane in &mut frustum.planes {
            plane.normalize();
        }

        frustum
    }

    pub fn contains(&self, position: Vec3, radius: f32) -> bool {
        self.planes.iter().all(|plane| plane.distance_to_point(&position) >= -radius)
    }
}
