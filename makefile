OBJS = D_update.o D_update_pml.o E_update.o H_update.o H_update_pml.o PML_class.o PML_field_initialize.o \
		PML_idx_initialize.o add_J.o array_ini.o calc_fdtd.o main.o sigma_calc.o E_old_to_new.o Ne_allocate.o \
		ny_allocate.o set_matrix.o surface_impe_calc.o surface_H_update.o
HEADERS = fdtd3d.h main.h PML.h gcc.h
OPTS = -O3 -I/usr/include/eigen3 -I/usr/include -L/usr/lib -lnrlmsise
GCCOPTS = -I/usr/include/eigen3 -std=c++1z -O3

main: $(OBJS)
	nvcc -o $@ $(OBJS) $(OPTS)
%.o: %.cu $(HEADERS)
	nvcc -c $< $(OPTS)
%.o: %.cpp $(HEADERS)
	g++ -c $< $(GCCOPTS)