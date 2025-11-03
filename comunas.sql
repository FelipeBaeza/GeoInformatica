INSERT INTO comunas (id, nombre) VALUES
(1, 'Cerrillos'), (2, 'Cerro Navia'), (3, 'Conchalí'), (4, 'El Bosque'),
(5, 'Estación Central'), (6, 'Huechuraba'), (7, 'Independencia'), (8, 'La Cisterna'),
(9, 'La Florida'), (10, 'La Granja'), (11, 'La Pintana'), (12, 'La Reina'),
(13, 'Las Condes'), (14, 'Lo Barnechea'), (15, 'Lo Espejo'), (16, 'Lo Prado'),
(17, 'Macul'), (18, 'Maipú'), (19, 'Ñuñoa'), (20, 'Pedro Aguirre Cerda'),
(21, 'Peñalolén'), (22, 'Providencia'), (23, 'Pudahuel'), (24, 'Quilicura'),
(25, 'Quinta Normal'), (26, 'Recoleta'), (27, 'Renca'), (28, 'San Joaquín'),
(29, 'San Miguel'), (30, 'San Ramón'), (31, 'Santiago'), (32, 'Vitacura')
ON CONFLICT (id) DO NOTHING;
