CXX ?= g++
FLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -Iinclude
BUILD = ../build
all: $(BUILD)/ids_train $(BUILD)/libidsengine.so $(BUILD)/test_engine
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/ids_train: src/mlp.cpp src/train_main.cpp | $(BUILD)
	$(CXX) $(FLAGS) $^ -o $@
$(BUILD)/libidsengine.so: src/mlp.cpp src/c_api.cpp | $(BUILD)
	$(CXX) $(FLAGS) -fPIC -shared $^ -o $@
$(BUILD)/test_engine: src/mlp.cpp tests/test_engine.cpp | $(BUILD)
	$(CXX) -std=c++17 -O1 -Wall -Wextra -Iinclude -DIDS_REAL=double $^ -o $@
test: $(BUILD)/test_engine
	$(BUILD)/test_engine
clean:
	rm -rf $(BUILD)
