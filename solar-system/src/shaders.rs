use nalgebra_glm::{dot, mat4_to_mat3, normalize, Mat3, Vec2, Vec3, Vec4};
use crate::vertex::Vertex;
use crate::Uniforms;
use crate::fragment::Fragment;
use crate::color::Color;


pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Vertex {
  // Transform position
  let position = Vec4::new(
    vertex.position.x,
    vertex.position.y,
    vertex.position.z,
    1.0
  );
  let transformed = uniforms.projection_matrix * uniforms.view_matrix * uniforms.model_matrix * position;

  // Perform perspective division
  let w = transformed.w;
  let ndc_position = Vec4::new(
    transformed.x / w,
    transformed.y / w,
    transformed.z / w,
    1.0
  );

  // apply viewport matrix
  let screen_position = uniforms.viewport_matrix * ndc_position;

  // Transform normal
  let model_mat3 = mat4_to_mat3(&uniforms.model_matrix); 
  let normal_matrix = model_mat3.transpose().try_inverse().unwrap_or(Mat3::identity());

  let transformed_normal = normal_matrix * vertex.normal;

  // Create a new Vertex with transformed attributes
  Vertex {
    position: vertex.position,
    normal: vertex.normal,
    tex_coords: vertex.tex_coords,
    color: vertex.color,
    transformed_position: Vec3::new(screen_position.x, screen_position.y, screen_position.z),
    transformed_normal,
  }
}

pub fn fragment_shader(fragment: &Fragment, uniforms: &Uniforms, time: u32) -> (Color, u32) {
  match uniforms.current_shader {
      1 => earth_shader(fragment, uniforms, time),
      2 => mars_planet_shader(fragment, uniforms),
      3 => mercury_shader(fragment, uniforms),
      4 => saturn_shader(fragment),
      5 => jupiter_shader(fragment, uniforms),
      6 => urano_shader(fragment, uniforms, time),
      7 => sun_shader(),
      8 => moon_shader(fragment, uniforms),
      9 => ring_shader(fragment),
      _ => (Color::new(0, 0, 0), 0), // Color por defecto si no hay un shader definido
  }
}

fn earth_shader(fragment: &Fragment, uniforms: &Uniforms, time: u32) -> (Color, u32) {
  let zoom = 100.0;  // to move our values 
  let ox = 100.0; // offset x in the noise map
  let oy = 100.0;
  let x = fragment.vertex_position.x;
  let y = fragment.vertex_position.y;
  let t = time as f32 * 0.1;

  let base_noise_value = uniforms.noise.get_noise_2d(x, y);
  let cloud_noise_value = uniforms.cloud_noise.get_noise_2d(
      x * zoom + ox +t, y * zoom + oy
  );

  // Colores base para el agua y la tierra
  let water_color_1 = Color::from_float(0.0, 0.1, 0.6); // Azul oscuro
  let water_color_2 = Color::from_float(0.0, 0.3, 0.7); // Azul claro
  let land_color_1 = Color::from_float(0.1, 0.5, 0.0); // Verde oscuro
  let land_color_2 = Color::from_float(0.2, 0.8, 0.2); // Verde claro
  let cloud_color = Color::from_float(0.9, 0.9, 0.9); // Color casi blanco para las nubes

  let land_threshold = 0.3; // Umbral para tierra

  // Decidir si el fragmento es agua o tierra
  let base_color = if base_noise_value > land_threshold {
      land_color_1.lerp(&land_color_2, (base_noise_value - land_threshold) / (1.0 - land_threshold))
  } else {
      water_color_1.lerp(&water_color_2, base_noise_value / land_threshold)
  };

  // Iluminación más dramática
  let light_position = Vec3::new(1.0, 1.0, 3.0); // Posición de la luz ajustada para mayor contraste
  let light_dir = normalize(&(light_position - fragment.vertex_position)); // Dirección de la luz ajustada
  let normal = normalize(&fragment.normal); // Normalizar la normal
  let diffuse = dot(&normal, &light_dir).max(0.0); // Cálculo de la componente difusa

  let lit_color = base_color * (0.1 + 0.9 * diffuse); 

  let cloud_threshold = 0.1; // Umbral para la aparición de nubes
  let cloud_opacity = 0.3 + 0.2 * ((time as f32 / 1000.0) * 0.3).sin().abs(); 
  if cloud_noise_value > cloud_threshold {
      let cloud_intensity = ((cloud_noise_value - cloud_threshold) / (1.0 - cloud_threshold)).clamp(0.0, 1.0);
      (lit_color.blend_add(&(cloud_color * (cloud_intensity * cloud_opacity))), 0)
  } else {
      (lit_color, 0)
  }
}

fn mars_planet_shader(fragment: &Fragment, uniforms: &Uniforms) -> (Color, u32) {
  let noise_value = uniforms.noise.get_noise_2d(fragment.vertex_position.x, fragment.vertex_position.y);
  
  let dark_red = Color::from_float(0.4, 0.1, 0.1); // Color oscuro para áreas en sombra
  let bright_orange = Color::from_float(0.8, 0.4, 0.1); // Color brillante para áreas iluminadas
  let terracotta = Color::from_float(0.6, 0.3, 0.1); // Color intermedio, típico de Marte

  // Usar lerp para mezclar colores basado en el valor del ruido
  let lerp_factor = noise_value.clamp(0.0, 1.0); // Asegurar que esté entre 0 y 1
  let base_color = if lerp_factor < 0.5 {
    dark_red.lerp(&terracotta, lerp_factor * 2.0) // Interpola entre rojo oscuro y terracotta
  } else {
    terracotta.lerp(&bright_orange, (lerp_factor - 0.5) * 2.0) // Interpola entre terracotta y naranja brillante
  };

  // Definir la posición y dirección de la luz
  let light_pos = Vec3::new(0.0, 8.0, 9.0);  // Posición de la fuente de luz
  let light_dir = (light_pos - fragment.vertex_position).normalize(); // Dirección de la luz desde la posición del fragmento

  // Normalizar la normal del fragmento
  let normal = fragment.normal.normalize();

  // Calcular la intensidad de la luz difusa
  let diffuse_intensity = normal.dot(&light_dir).max(0.0);

  // Modificar el color final basado en la intensidad de la luz
  let lit_color = base_color * diffuse_intensity;  // Modula el color por la intensidad de la luz

  // Añadir un término ambiental para evitar que las partes no iluminadas sean completamente oscuras
  let ambient_intensity = 0.15;  // Intensidad de luz ambiental, ajusta según necesites
  let ambient_color = base_color * ambient_intensity;

  // Suma del componente ambiental y difuso
  let combined_color = ambient_color + lit_color;

  (combined_color, 0)
}

pub fn moon_shader(fragment: &Fragment, uniforms: &Uniforms) -> (Color, u32) {
  // Base y detalles de color más distintos
  let base_color = Color::from_float(0.8, 0.8, 0.8); // Gris base
  let detail_color = Color::from_float(0.3, 0.3, 0.3); // Gris más oscuro para detalles

  // Genera variaciones en la superficie
  let noise_value = uniforms.noise.get_noise_2d(fragment.vertex_position.x, fragment.vertex_position.y);

  // Normaliza el valor del ruido a [0, 1]
  let normalized_noise = (noise_value + 1.0) * 0.5; // Ajusta según el rango real de tu generador de ruido
  let surface_variation = base_color.lerp(&detail_color, normalized_noise.clamp(0.0, 1.0));

  // Iluminación simple
  let light_position = Vec3::new(10.0, 10.0, 10.0);
  let light_direction = (light_position - fragment.vertex_position).normalize();
  let normal = fragment.normal.normalize();
  let diffuse = normal.dot(&light_direction).max(0.0);

  // Combinar color de superficie con iluminación
  (surface_variation * (0.3 + 0.7 * diffuse), 0)
}

pub fn mercury_shader(fragment: &Fragment, uniforms: &Uniforms) -> (Color, u32) {
  // Colores base para la superficie de Mercurio
  let gray_light = Color::from_float(0.7, 0.7, 0.7);
  let gray_dark = Color::from_float(0.4, 0.4, 0.4);
  let brown = Color::from_float(0.5, 0.4, 0.3);
  let blue_tint = Color::from_float(0.3, 0.3, 0.7);
  let yellow_light = Color::from_float(0.8, 0.7, 0.4);

  // Genera ruido para variaciones de color
  let noise_value1 = uniforms.noise.get_noise_2d(fragment.vertex_position.x, fragment.vertex_position.y);
  let noise_value2 = uniforms.noise.get_noise_2d(fragment.vertex_position.x * 2.0, fragment.vertex_position.y * 2.0); // Ajustar frecuencia
  let noise_value3 = uniforms.noise.get_noise_2d(fragment.vertex_position.x * 0.5, fragment.vertex_position.y * 0.5); // Baja frecuencia

  // Normaliza los valores de ruido
  let lerp_factor1 = (noise_value1 + 1.0) * 0.5; // Normalizar a [0, 1]
  let lerp_factor2 = (noise_value2 + 1.0) * 0.5;
  let lerp_factor3 = (noise_value3 + 1.0) * 0.5;

  // Mezcla de colores usando `lerp`
  let color_mix1 = gray_light.lerp(&gray_dark, lerp_factor1);
  let color_mix2 = color_mix1.lerp(&brown, lerp_factor2 * 2.5);
  let color_mix3 = color_mix2.lerp(&blue_tint, lerp_factor2 * 1.5);
  let final_color = color_mix3.lerp(&yellow_light, lerp_factor3);

  // Iluminación para dar más realismo
  let light_position = Vec3::new(0.0, 8.0, 9.0);
  let light_direction = (light_position - fragment.vertex_position).normalize();
  let normal = fragment.normal.normalize();
  let diffuse = normal.dot(&light_direction).max(0.0);

  // Combinación de la iluminación con el color
  let ambient_intensity = 0.15;
  let ambient_color = final_color * ambient_intensity;
  let lit_color = final_color * diffuse;

  // Suma del componente ambiental y difuso
  (ambient_color + lit_color, 0)
}

// Tranisicón suave de la Gran Mancha Roja
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
  t * t * (3.0 - 2.0 * t)
}
// Calculo de uv 
fn calculate_uv(position: Vec3) -> Vec2 {
  let theta = position.z.atan2(position.x); // Longitud
  let phi = position.y.asin(); // Latitud
  Vec2::new(
      0.5 + theta / (2.0 * std::f32::consts::PI),
      0.5 - phi / std::f32::consts::PI,
  )
}    

fn jupiter_shader(fragment: &Fragment, uniforms: &Uniforms) -> (Color, u32) {
  // Capa 1: Bandas horizontales difuminadas
  let latitude = fragment.vertex_position.y;
  let band_frequency = 10.0;

  // Agregar ruido al patrón de bandas
  let band_noise = uniforms.band_noise.get_noise_2d(
      fragment.vertex_position.x * 2.0,
      fragment.vertex_position.y * 2.0,
  );
  let band_noise_intensity = 0.2;
  let distorted_latitude = latitude + band_noise * band_noise_intensity;
  let band_pattern = (distorted_latitude * band_frequency).sin();

  // Definir una paleta de colores con los nuevos tonos
  let band_colors = [
      Color::from_hex(0xc6bcad), // Color base
      Color::from_hex(0x955d36), // Color de las líneas
      Color::from_hex(0xc7c7cf), // Reemplazo del azul con gris claro
  ];

  // Interpolación suave entre colores
  let normalized_band = (band_pattern + 1.0) / 2.0 * (band_colors.len() as f32 - 1.0);
  let index = normalized_band.floor() as usize;
  let t = normalized_band.fract();
  let color1 = band_colors[index % band_colors.len()];
  let color2 = band_colors[(index + 1) % band_colors.len()];
  let base_color = color1.lerp(&color2, t);

  // Capa 2: Turbulencia con ruido
  let noise_value = uniforms.noise.get_noise_3d(
      fragment.vertex_position.x * 4.0,
      fragment.vertex_position.y * 4.0,
      fragment.vertex_position.z * 4.0,
  );

  let turbulence_intensity = 0.3;
  let turbulence_color = base_color.lerp(&Color::from_hex(0xffffff), noise_value * turbulence_intensity);

  // Capa adicional: Variación de color con ruido
  let color_noise_value = uniforms.noise.get_noise_3d(
      fragment.vertex_position.x * 3.0,
      fragment.vertex_position.y * 3.0,
      fragment.vertex_position.z * 3.0,
  );

  let terracotta_variation_color = Color::from_hex(0x955d36); // Color terracota para variaciones
  let gray_variation_color = Color::from_hex(0xc7c7cf); // Gris claro para variaciones

  let terracotta_intensity = ((color_noise_value + 1.0) * 0.5).clamp(0.0, 1.0);
  let gray_intensity = (1.0 - terracotta_intensity).clamp(0.0, 1.0);

  let color_with_variation = turbulence_color
      .lerp(&terracotta_variation_color, terracotta_intensity * 0.2)
      .lerp(&gray_variation_color, gray_intensity * 0.2);

  // Capa 3: Gran Mancha Roja con color ajustado y difuminada
  let uv = fragment.uv.unwrap_or_else(|| calculate_uv(fragment.vertex_position));
  let red_spot_center = Vec2::new(0.65, 0.5);
  let distance_to_spot = (uv - red_spot_center).norm();

  let red_spot_noise_value = uniforms.noise.get_noise_2d(
      uv.x * 20.0,
      uv.y * 20.0,
  );
  let red_spot_noise_intensity = red_spot_noise_value * 0.3;

  let red_spot_radius = 0.1;
  let red_spot_edge = 0.08;
  let red_spot_intensity = smoothstep(
      red_spot_radius + red_spot_edge,
      red_spot_radius - red_spot_edge,
      distance_to_spot,
  );

  let red_spot_intensity = (red_spot_intensity + red_spot_noise_intensity).clamp(0.0, 1.0);

  let red_spot_color = Color::from_hex(0xac6300); // Nuevo color de la Gran Mancha Roja
  let final_color = color_with_variation.lerp(&red_spot_color, red_spot_intensity * 0.9);

  // Iluminación
  let light_position = Vec3::new(0.0, 8.0, 9.0);
  let light_direction = (light_position - fragment.vertex_position).normalize();
  let normal = fragment.normal.normalize();
  let diffuse = normal.dot(&light_direction).max(0.0);

  // Combinación de la iluminación con el color
  let ambient_intensity = 0.15;
  let ambient_color = final_color * ambient_intensity;
  let lit_color = final_color * diffuse;

  // Suma del componente ambiental y difuso
  let color_with_lighting = ambient_color + lit_color;

  (color_with_lighting, 0)
}

fn urano_shader(fragment: &Fragment, uniforms: &Uniforms, time: u32) -> (Color, u32) {
  let x = fragment.vertex_position.x;
  let y = fragment.vertex_position.y;
  let z = fragment.vertex_position.z;
  let t = time as f32 * 0.001; // Escala de tiempo para el movimiento

  // Coordenadas de ruido para simular movimiento atmosférico
  let noise_value = uniforms.noise.get_noise_3d(x, y + t, z);

  // Color base de Urano
  let base_color = Color::from_float(0.2, 0.5, 0.9); // Un azul característico de Urano

  // Intensidad del ruido para variar el color base
  let intensity = (noise_value * 0.5 + 0.5).clamp(0.0, 1.0); // Normaliza y asegura los límites
  let varied_color = base_color * intensity;

  // Iluminación direccional para resaltar la textura
  let light_dir = Vec3::new(1.0, 1.0, 1.0).normalize(); // Dirección de luz arbitraria
  let normal = fragment.normal.normalize(); // Normalizar la normal del fragmento
  let diffuse = normal.dot(&light_dir).max(0.0); // Cálculo difuso
  let ambient = 0.3; // Intensidad ambiental
  let lit_color = varied_color * (ambient + (1.0 - ambient) * diffuse); // Combinación de iluminación

  (lit_color, 0)
}

fn saturn_shader(fragment: &Fragment) -> (Color, u32) {
  // Normalizar la latitud de -1 a 1 a un rango de 0 a 1
  let latitude = (fragment.vertex_position.y + 1.0) * 0.5;

  // Gradiente de colores desde el polo hasta el ecuador
  let colors = [
      Color::from_hex(0x6b6255), // Oscuro en las puntas
      Color::from_hex(0xe0cdaf), // Color de transición hacia el centro
      Color::from_hex(0xe8d4ab), // Color central claro
      Color::from_hex(0xcfb98c), // Segundo color de transición
      Color::from_hex(0xfef3d1), // Color central más claro
      Color::from_hex(0xcfb98c), // Repetir para simetría
      Color::from_hex(0xe8d4ab),
      Color::from_hex(0xe0cdaf),
      Color::from_hex(0x6b6255)  // Oscuro en las puntas
  ];

  // Calcular la posición en el array de colores basado en la latitud
  let position_in_gradient = latitude * (colors.len() - 1) as f32;
  let index = position_in_gradient.floor() as usize;
  let frac = position_in_gradient.fract();

  // Lerp entre colores cercanos para suavizar el gradiente
  let base_color = colors[index];
  let next_color = colors[index + 1 % colors.len()];
  let color = base_color.lerp(&next_color, frac);

  // Aplicar iluminación básica
  let light_position = Vec3::new(1.0, 1.0, 10.0);
  let light_direction = (light_position - fragment.vertex_position).normalize();
  let normal = fragment.normal.normalize();
  let diffuse = normal.dot(&light_direction).max(0.0);

  let ambient_intensity = 0.1;  // Ajustar según la escena
  let ambient_color = color * ambient_intensity;
  let diffuse_color = color * diffuse;

  (ambient_color + diffuse_color, 0)
}

pub fn ring_shader(fragment: &Fragment) -> (Color, u32) {
  // Coordenadas en 2D para determinar la distancia desde el centro de los anillos
  let position = Vec2::new(fragment.vertex_position.x, fragment.vertex_position.z); // Usar X y Z para planos
  let distance_from_center = position.magnitude(); // Calcular la distancia desde el centro

  // Definir el número de bandas y su ancho
  let num_bands = 2; // Número total de bandas en los anillos
  let max_distance = 1.0; // Distancia máxima para las bandas (ajustar según el tamaño de los anillos)
  let band_width = max_distance / num_bands as f32; // Ancho de cada banda

  // Calcular en qué banda está el fragmento actual
  let band_index = (distance_from_center / band_width).floor() as i32;

  // Variar el color de los anillos en función de su índice
  let band_colors = [
      Color::from_hex(0x817970), // Gris claro
      Color::from_hex(0x474744), // Gris oscuro
      Color::from_hex(0x817970), // Gris claro
      Color::from_hex(0x474744), // Gris oscuro
  ];

  // Seleccionar el color basado en el índice de la banda y el número de bandas
  let color = band_colors[(band_index.abs() % num_bands) as usize % band_colors.len()];

  // Aplicar un efecto de difuminado en los bordes de las bandas
  let edge_distance = (distance_from_center % band_width) / band_width;
  let smooth_edge = (1.0 - edge_distance).clamp(0.0, 1.0);

  // Modificar la opacidad para dar un efecto de transparencia a los anillos
  let final_color = color * smooth_edge;

  (final_color, 0)
}

fn sun_shader() -> (Color, u32) {
  let base_color = Color::from_float(1.0, 0.9, 0.5); // Color amarillo/dorado para el Sol
  let emission = 100; // Máxima emisión para el efecto de glow/bloom

  (base_color, emission)
}
