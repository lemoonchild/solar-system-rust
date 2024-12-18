use nalgebra_glm::{Vec3, Mat4, look_at, perspective};
use minifb::{Key, Window, WindowOptions};
use std::{f32::consts::PI, time::Instant};

mod framebuffer;
mod triangle;
mod vertex;
mod obj;
mod color;
mod fragment;
mod shaders;
mod camera;
mod skybox;
mod frustum;

use framebuffer::Framebuffer;
use vertex::Vertex;
use obj::Obj;
use camera::Camera;
use skybox::Skybox; 
use frustum::{Frustum, Plane};
use triangle::triangle;
use shaders::{vertex_shader, fragment_shader};
use fastnoise_lite::{FastNoiseLite, NoiseType, FractalType};


pub struct Uniforms {
    model_matrix: Mat4,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    viewport_matrix: Mat4,
    time: u32,
    noise: FastNoiseLite,
    cloud_noise: FastNoiseLite, 
    band_noise: FastNoiseLite, 
    current_shader: u8, 
}

struct AppState {
    last_mouse_pos: Option<(f32, f32)>,
    bird_eye_active: bool,
}

fn create_noise(current_shader: u8) -> FastNoiseLite {
    match current_shader {
        1 => create_earth_noise(),
        2 => create_mars_noise(),
        3 => create_mercury_noise(),
        4 => FastNoiseLite::new(),
        5 => create_jupiter_noise(),
        6 => create_urano_noise(), 
        8 => create_moon_noise(),
        9 => FastNoiseLite::new(),
        10 => create_spaceship_noise(),
        _ => create_earth_noise(),  
    }
}

fn create_earth_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(1337);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2S));
    noise.set_fractal_type(Some(FractalType::Ridged));
    noise.set_fractal_octaves(Some(5)); // Octavas para mayor detalle
    noise.set_fractal_lacunarity(Some(3.0)); // Lacunaridad para escalado de frecuencia
    noise.set_fractal_gain(Some(0.5)); // Ganancia para el escalado de amplitud
    noise.set_frequency(Some(0.5)); 

    noise
}

fn create_cloud_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(40);  
    noise.set_noise_type(Some(NoiseType::Perlin)); 
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(2));  // Menos octavas para menos detalles
    noise.set_fractal_lacunarity(Some(3.0));
    noise.set_fractal_gain(Some(0.5));
    noise.set_frequency(Some(0.01));  // Baja frecuencia para nubes grandes y suaves
    noise
}

fn create_mars_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(1234);
    noise.set_noise_type(Some(NoiseType::Perlin));
    noise.set_fractal_type(Some(FractalType::Ridged));
    noise.set_fractal_octaves(Some(4));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.5));
    noise.set_frequency(Some(1.5)); 
    noise
}

fn create_moon_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(4321);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_fractal_type(Some(FractalType::PingPong));
    noise.set_fractal_octaves(Some(2));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.5));
    noise.set_frequency(Some(3.0));  
    noise
}

fn create_mercury_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(4321);
    noise.set_noise_type(Some(NoiseType::Perlin));
    noise.set_fractal_type(Some(FractalType::PingPong));
    noise.set_fractal_octaves(Some(5));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(1.0));
    noise.set_frequency(Some(5.0));  
    noise
}

fn create_jupiter_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(5678); // Puedes elegir cualquier semilla
    noise.set_noise_type(Some(NoiseType::OpenSimplex2)); // OpenSimplex2 produce un ruido más suave
    noise.set_fractal_type(Some(FractalType::DomainWarpProgressive)); // Añade complejidad fractal
    noise.set_fractal_octaves(Some(6)); // Más octavas para más detalle
    noise.set_fractal_lacunarity(Some(2.0)); // Lacunaridad estándar
    noise.set_fractal_gain(Some(0.5)); // Ganancia menor para detalles finos
    noise.set_frequency(Some(2.0)); // Ajusta la escala del ruido
    noise
}

fn create_jupiter_band_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(7890); // Nueva semilla
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_frequency(Some(1.0));
    noise.set_fractal_type(Some(FractalType::FBm));
    noise
}

fn create_urano_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(2021);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_fractal_type(Some(FractalType::Ridged));
    noise.set_fractal_octaves(Some(4));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.4));
    noise.set_frequency(Some(0.2));
    noise
}


fn create_spaceship_noise() -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(2021);
    noise.set_noise_type(Some(NoiseType::Perlin));
    noise.set_fractal_type(Some(FractalType::Ridged));
    noise.set_fractal_octaves(Some(4));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.4));
    noise.set_frequency(Some(0.5));
    noise
}

fn create_model_matrix(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    let rotation_matrix_x = Mat4::new(
        1.0,  0.0,    0.0,   0.0,
        0.0,  cos_x, -sin_x, 0.0,
        0.0,  sin_x,  cos_x, 0.0,
        0.0,  0.0,    0.0,   1.0,
    );

    let rotation_matrix_y = Mat4::new(
        cos_y,  0.0,  sin_y, 0.0,
        0.0,    1.0,  0.0,   0.0,
        -sin_y, 0.0,  cos_y, 0.0,
        0.0,    0.0,  0.0,   1.0,
    );

    let rotation_matrix_z = Mat4::new(
        cos_z, -sin_z, 0.0, 0.0,
        sin_z,  cos_z, 0.0, 0.0,
        0.0,    0.0,  1.0, 0.0,
        0.0,    0.0,  0.0, 1.0,
    );

    let rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    let transform_matrix = Mat4::new(
        scale, 0.0,   0.0,   translation.x,
        0.0,   scale, 0.0,   translation.y,
        0.0,   0.0,   scale, translation.z,
        0.0,   0.0,   0.0,   1.0,
    );

    transform_matrix * rotation_matrix
}


fn create_view_matrix(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    look_at(&eye, &center, &up)
}

fn create_perspective_matrix(window_width: f32, window_height: f32) -> Mat4 {
    let fov = 45.0 * PI / 180.0;
    let aspect_ratio = window_width / window_height;
    let near = 0.1;
    let far = 1000.0;

    perspective(fov, aspect_ratio, near, far)
}

fn create_viewport_matrix(width: f32, height: f32) -> Mat4 {
    Mat4::new(
        width / 2.0, 0.0, 0.0, width / 2.0,
        0.0, -height / 2.0, 0.0, height / 2.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
}

fn gaussian_blur(buffer: &mut [u32], width: usize, height: usize, kernel_size: usize, sigma: f32) {
    let gaussian_kernel = create_gaussian_kernel(kernel_size, sigma);
    let kernel_sum: f32 = gaussian_kernel.iter().map(|&x| x as f32).sum();

    // Aplicar horizontalmente
    for y in 0..height {
        let mut temp_row = vec![0u32; width];
        for x in 0..width {
            let mut filtered_pixel = 0f32;
            for k in 0..gaussian_kernel.len() {
                let sample_x = x as i32 + k as i32 - (gaussian_kernel.len() / 2) as i32;
                if sample_x >= 0 && sample_x < width as i32 {
                    filtered_pixel += buffer[sample_x as usize + y * width] as f32 * gaussian_kernel[k] as f32;
                }
            }
            temp_row[x] = (filtered_pixel / kernel_sum).round() as u32;
        }
        buffer[y * width..(y + 1) * width].copy_from_slice(&temp_row);
    }

    // Aplicar verticalmente
    for x in 0..width {
        let mut temp_col = vec![0u32; height];
        for y in 0..height {
            let mut filtered_pixel = 0f32;
            for k in 0..gaussian_kernel.len() {
                let sample_y = y as i32 + k as i32 - (gaussian_kernel.len() / 2) as i32;
                if sample_y >= 0 && sample_y < height as i32 {
                    filtered_pixel += buffer[x + sample_y as usize * width] as f32 * gaussian_kernel[k] as f32;
                }
            }
            temp_col[y] = (filtered_pixel / kernel_sum).round() as u32;
        }
        for y in 0..height {
            buffer[x + y * width] = temp_col[y];
        }
    }
}

// Crear un kernel Gaussiano dinámicamente
fn create_gaussian_kernel(size: usize, sigma: f32) -> Vec<u32> {
    let mut kernel = vec![0u32; size];
    let mean = (size as f32 - 1.0) / 2.0;
    let coefficient = 1.0 / (2.0 * std::f32::consts::PI * sigma * sigma).sqrt();

    for x in 0..size {
        let exp_numerator = -((x as f32 - mean) * (x as f32 - mean)) / (2.0 * sigma * sigma);
        let exp_value = (-exp_numerator).exp();
        kernel[x] = (coefficient * exp_value * 255.0) as u32;
    }

    kernel
}

fn apply_bloom(original: &mut [u32], bloom: &[u32], width: usize, height: usize) {
    for i in 0..original.len() {
        let original_color = original[i];
        let bloom_intensity = bloom[i];
        if bloom_intensity > 0 {
            original[i] = blend_bloom(original_color, bloom_intensity);
        }
    }
}

fn blend_bloom(base_color: u32, bloom_intensity: u32) -> u32 {
    // Factores para el tonemapping y la mezcla de bloom
    let bloom_strength = 0.8;  // Ajusta esto para controlar la fuerza del efecto de bloom
    let max_bloom_effect = 1.2;  // Este valor limita cuánto puede influir el bloom

    let r = ((base_color >> 16) & 0xFF) as f32;
    let g = ((base_color >> 8) & 0xFF) as f32;
    let b = (base_color & 0xFF) as f32;
    let bloom = bloom_intensity as f32 * bloom_strength;

    // Calcular nueva intensidad de color con clamping para evitar saturación
    let new_r = ((r + bloom).min(255.0 * max_bloom_effect)).min(255.0) as u32;
    let new_g = ((g + bloom).min(255.0 * max_bloom_effect)).min(255.0) as u32;
    let new_b = ((b + bloom).min(255.0 * max_bloom_effect)).min(255.0) as u32;

    // Recomponer el color
    (new_r << 16) | (new_g << 8) | new_b
}

fn render(framebuffer: &mut Framebuffer, uniforms: &Uniforms, vertex_array: &[Vertex], time: u32, disable_culling: bool) {
    // Vertex Shader Stage
    let mut transformed_vertices = Vec::with_capacity(vertex_array.len());
    for vertex in vertex_array {
        let transformed = vertex_shader(vertex, uniforms);
        transformed_vertices.push(transformed);
    }

    // Primitive Assembly Stage
    let mut triangles = Vec::new();
    for i in (0..transformed_vertices.len()).step_by(3) {
        if i + 2 < transformed_vertices.len() {
            triangles.push([
                transformed_vertices[i].clone(),
                transformed_vertices[i + 1].clone(),
                transformed_vertices[i + 2].clone(),
            ]);
        }
    }

    // Rasterization Stage
    let mut fragments = Vec::new();
    for tri in &triangles {
        fragments.extend(triangle(&tri[0], &tri[1], &tri[2], disable_culling));
    }

    // Fragment Processing Stage
    for fragment in fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            // Apply fragment shader
            let (shaded_color, emission) = fragment_shader(&fragment, &uniforms, time);
            let color = shaded_color.to_hex();
            framebuffer.set_current_color(color);
            framebuffer.point(x, y, fragment.depth, emission);  // Asegúrate de que `point` acepte y maneje `emission`
        }
    }
}

fn check_collision(camera_pos: Vec3, planet_pos: Vec3, planet_radius: f32) -> bool {
    let distance = (camera_pos - planet_pos).magnitude();
    distance < (planet_radius + 25.0)  // margen de seguridad
}

fn create_ortho_projection(window_width: f32, window_height: f32) -> Mat4 {
    nalgebra_glm::ortho(0.0, window_width, window_height, 0.0, -1.0, 1.0)
}

fn main() {
    let window_width = 680;
    let window_height = 700;
    let framebuffer_width = 680;
    let framebuffer_height = 700;

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new(
        "Rust Graphics - Solar System Final Project",
        window_width,
        window_height,
        WindowOptions::default(),
    )
        .unwrap();

    window.set_position(500, 500);
    window.update();

    framebuffer.set_background_color(0x000000);

    // model position
    let rotation = Vec3::new(0.0, 0.0, 0.0);

    // camera parameters
    let mut camera = Camera::new(
        Vec3::new(0.0, 50.0, 60.0),
        Vec3::new(22.5, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0)
    );

    // Aquí inicializamos el estado
    let mut app_state = AppState {
        last_mouse_pos: None,
        bird_eye_active: false,
    };

    let obj = Obj::load("assets/models/sphere.obj").expect("Failed to load obj");
    let moon = Obj::load("assets/models/moon.obj").expect("Failed to load obj");
    let ring_obj = Obj::load("assets/models/ring.obj").expect("Failed to load ring model");
    let spaceship = Obj::load("assets/models/spaceship.obj").expect("Failed to load spaceship model");

    let skybox = Skybox::new(5000);

    let vertex_arrays = obj.get_vertex_array(); 
    let moon_vertex_array = moon.get_vertex_array();
    let ring_vertex_array = ring_obj.get_vertex_array();
    let spaceship_vertex_array = spaceship.get_vertex_array();

    let mut last_frame_time = Instant::now();
    let mut time = 0;

    // Lunas de los planetas rocosos
    let moon_scale = 0.5; // Escala de la luna respecto al planeta
    let moon_distance = 2.5; // Distancia de la luna al planeta
    let moon_orbit_speed = 0.001; // Velocidad orbital de la luna

    let projection_matrix = create_perspective_matrix(window_width as f32, window_height as f32);
    let viewport_matrix = create_viewport_matrix(framebuffer_width as f32, framebuffer_height as f32);
    let mut uniforms = Uniforms { 
        model_matrix: Mat4::identity(), 
        view_matrix: Mat4::identity(), 
        projection_matrix, 
        viewport_matrix, 
        time: 0, 
        noise: create_noise(1),
        cloud_noise: create_cloud_noise(),
        band_noise: create_jupiter_band_noise(), 
        current_shader: 1,
    };

    let planet_positions = vec![
        Vec3::new(0.0, 0.0, 0.0),  // Sol
        Vec3::new(10.0, 0.0, 0.0),  // Mercurio
        Vec3::new(15.0, 0.0, 0.0),  // Tierra
        Vec3::new(25.0, 0.0, 0.0),  // Marte
        Vec3::new(35.0, 0.0, 0.0), // Júpiter
        Vec3::new(45.0, 0.0, 0.0), // Saturno
        Vec3::new(60.0, 0.0, 0.0)  // Urano
    ];

    let scales = vec![
        2.0,  // Escala para el Sol (más grande)
        0.5,  // Escala para Mercurio
        0.8,  // Escala para la Tierra
        0.5,  // Escala para Marte
        1.5,  // Escala para Júpiter
        0.8,  // Escala para Saturno
        0.6   // Escala para Urano
    ];

    let orbital_speeds = vec![
        0.0,   // Sol (estático)
        0.5,  // Mercurio
        0.2,  // Tierra
        0.1,  // Marte
        0.05,  // Júpiter
        0.03, // Saturno
        0.01  // Urano
    ]; 

    // Definimos los shaders de cada planeta en el orden correcto
    let shaders = vec![7, 3, 1, 2, 5, 4, 6];
    let mut orbital_angles = vec![0.0; shaders.len()]; 

    let system_center = Vec3::new(0.0, 0.0, 0.0);
    let mut bird_eye_active = false; 

    let hud_camera_projection = create_ortho_projection(window_width as f32, window_height as f32);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let delta_time = last_frame_time.elapsed();
        last_frame_time = Instant::now();
        time += delta_time.as_millis() as u32;
    
        handle_input(&window, &mut camera, system_center, &mut bird_eye_active, &mut app_state);
        framebuffer.clear();

        
        // Calcula matrices de vista y proyección para el sistema solar
        uniforms.projection_matrix = create_perspective_matrix(window_width as f32, window_height as f32);
        uniforms.view_matrix = create_view_matrix(camera.eye, camera.center, camera.up);
        let vp_matrix = uniforms.projection_matrix * uniforms.view_matrix;
        let frustum = Frustum::from_matrix(&vp_matrix);

        // Verifica las colisiones antes de actualizar la posición de la cámara y renderizar los objetos
        for (planet_pos, planet_radius) in planet_positions.iter().zip(scales.iter()) {
            if check_collision(camera.eye, *planet_pos, *planet_radius) {
                camera.prevent_collision(*planet_pos, *planet_radius);
            }
        }

        skybox.render(&mut framebuffer, &uniforms, camera.eye);

        // Matriz de visión siempre actualizada
        uniforms.view_matrix = create_view_matrix(camera.eye, camera.center, camera.up);
        uniforms.time = time as u32;

        // Actualiza y calcula las posiciones basadas en las órbitas
        for (i, original_position) in planet_positions.iter().enumerate() {
            // Actualizar el ángulo orbital
            orbital_angles[i] += orbital_speeds[i] * delta_time.as_secs_f32(); // Asumiendo que orbital_speeds está definido

            // Calcular la nueva posición usando coordenadas polares
            let orbit_radius = original_position.x;  // Asumiendo que x es el radio de la órbita
            let x = orbit_radius * orbital_angles[i].cos();
            let z = orbit_radius * orbital_angles[i].sin();
            let position = Vec3::new(x, 0.0, z); // Asumiendo una órbita plana en el plano xz

            let scale = scales[i];  // Usa la escala apropiada para cada planeta

            // Comprueba si el planeta está dentro del frustum antes de renderizar
            if frustum.contains(position, scale) {
                uniforms.current_shader = shaders[i];
                uniforms.model_matrix = create_model_matrix(position, scale, rotation);

                // Renderizado de planetas
                render(&mut framebuffer, &uniforms, &vertex_arrays, time as u32, false);

                // Renderizado condicional de lunas y anillos
                if shaders[i] == 2 {  // Marte con luna
                    let moon_angle = time as f32 * moon_orbit_speed;
                    let moon_x = moon_distance * moon_angle.cos();
                    let moon_z = moon_distance * moon_angle.sin();
                    let moon_translation = Vec3::new(moon_x, 0.0, moon_z) + position;

                    if frustum.contains(moon_translation, moon_scale) {
                        uniforms.current_shader = 8;
                        uniforms.model_matrix = create_model_matrix(moon_translation, moon_scale, Vec3::new(0.0, 0.0, 0.0));
                        render(&mut framebuffer, &uniforms, &moon_vertex_array, time as u32, false);
                    }
                } else if shaders[i] == 4 {  // Saturno con anillos
                    let ring_scale = scale * 1.5;
                    let ring_position = position;  // La posición de los anillos es la misma que la de Saturno

                    uniforms.current_shader = 9;
                    uniforms.model_matrix = create_model_matrix(ring_position, ring_scale, Vec3::new(0.0, 0.0, 0.0));
                    render(&mut framebuffer, &uniforms, &ring_vertex_array, time as u32, true);
                   
                } else if shaders[i] == 7 { // Sol con efecto Bloom
                    // Renderizar el Sol con efectos adicionales
                    render(&mut framebuffer, &uniforms, &vertex_arrays, time as u32, false);
                    let kernel_size = 10;
                    let sigma = 2.5;
                    gaussian_blur(&mut framebuffer.emissive_buffer, framebuffer.width, framebuffer.height, kernel_size, sigma);
                    apply_bloom(&mut framebuffer.buffer, &framebuffer.emissive_buffer, framebuffer.width, framebuffer.height);
                }
            }
        }

        // Cambia a la cámara ortográfica para renderizar la nave espacial en HUD
        uniforms.projection_matrix = hud_camera_projection;
        uniforms.view_matrix = Mat4::identity();
        uniforms.model_matrix = create_model_matrix(Vec3::new(window_width as f32 / 2.0, window_height as f32 - 100.0, 0.0), 15.0, Vec3::new(0.0, 0.0, 0.0));

        uniforms.current_shader = 10;
        // Renderiza la nave espacial en el HUD
        render(&mut framebuffer, &uniforms, &spaceship_vertex_array, time as u32, false);

    
        framebuffer.set_current_color(0xFFDDDD);
        window.update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height).unwrap();
    }    
}

fn handle_input(window: &Window, camera: &mut Camera, system_center: Vec3, bird_eye_active: &mut bool, app_state: &mut AppState) {
    let movement_speed = 1.0;
    let rotation_speed = PI/50.0;
    let zoom_speed = 1.5;
    let min_zoom_distance = 25.0; 

    if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(minifb::MouseMode::Pass) {
        if let Some((last_x, last_y)) = app_state.last_mouse_pos {
            let delta_x = mouse_x - last_x;
            let delta_y = mouse_y - last_y;
            camera.orbit(delta_x as f32 * rotation_speed, -delta_y as f32 * rotation_speed);
        }
        app_state.last_mouse_pos = Some((mouse_x, mouse_y));
    }

    let scroll = window.get_scroll_wheel();
    if let Some((_, scroll_y)) = scroll {
        let delta = scroll_y as f32 * zoom_speed;
        camera.zoom(delta, min_zoom_distance);
    }

    //  camera orbit controls
    if window.is_key_down(Key::Left) {
        camera.orbit(rotation_speed, 0.0);
    }
    if window.is_key_down(Key::Right) {
        camera.orbit(-rotation_speed, 0.0);
    }
    if window.is_key_down(Key::W) {
        camera.orbit(0.0, -rotation_speed);
    }
    if window.is_key_down(Key::S) {
        camera.orbit(0.0, rotation_speed);
    }

    // Camera movement controls
    let mut movement = Vec3::new(0.0, 0.0, 0.0);
    if window.is_key_down(Key::A) {
        movement.x -= movement_speed;
    }
    if window.is_key_down(Key::D) {
        movement.x += movement_speed;
    }
    if window.is_key_down(Key::Q) {
        movement.y += movement_speed;
    }
    if window.is_key_down(Key::E) {
        movement.y -= movement_speed;
    }
    if movement.magnitude() > 0.0 {
        camera.move_center(movement);
    }

    // Camera zoom controls
    if window.is_key_down(Key::Up) {
        camera.zoom(zoom_speed, min_zoom_distance);
    }
    if window.is_key_down(Key::Down) {
        camera.zoom(-zoom_speed, min_zoom_distance);
    }

    // Cambio a vista aérea
    // Alternar entre vista aérea y normal
     if window.is_key_pressed(Key::B, minifb::KeyRepeat::No) {
        if *bird_eye_active {
            // Resetear la cámara a la posición y orientación normal
            camera.eye = Vec3::new(0.0, 50.0, 40.0);
            camera.center = Vec3::new(22.5, 0.0, 0.0);
            camera.up = Vec3::new(0.0, 1.0, 0.0);
            *bird_eye_active = false;
        } else {
            // Cambiar a vista aérea
            camera.bird_eye_view(system_center, 100.0);
            *bird_eye_active = true;
        }
        camera.has_changed = true;
    }
}