CC = gcc
CFLAGS = -O2 -std=c99
MATHFLAG = -lm

all: raytracer_sequential

raytracer_sequential: main.c lib/vec_ops.c lib/geometry/Light.c lib/geometry/Light.h lib/geometry/ray.h lib/geometry/Sphere.c lib/geometry/Sphere.h lib/vec_ops.c
	$(CC) $(CFLAGS) -o raytracer_sequential main.c lib/geometry/Sphere.c lib/vec_ops.c lib/geometry/Light.c $(MATHFLAG)

run: raytracer_sequential
	./raytracer_sequential

clean: 
	rm -f raytracer_sequential