# Remove build_our if it exists
if [ -d "build_our" ]; then
    rm -rf build_our
fi

# Create build_our directory
mkdir build_our
cd build_our

# Run cmake to configure the project
cmake ..

# Build the project using make
make -j32 all

# Run the executable and test the performance
./examples/amgx_capi -m ../test_mat/test.mtx -c ../src/configs/PCG_V.json
./examples/amgx_capi -m ../test_mat/test.mtx -c ../src/configs/PCG_V.json > ../result.log