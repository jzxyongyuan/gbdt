WORKROOT=../
GLOG_PATH=$(WORKROOT)/glog
CONF_PATH=$(WORKROOT)/configure

INCLUDEDIR= -I$(GLOG_PATH)/include/ \
			-I$(CONF_PATH)/output/include \
			-I./inc

LIBDIR=		-L$(GLOG_PATH)/lib -lglog \
			-L$(CONF_PATH)/output/lib/ -lconfig\
			-lm \
			-lcrypto \
			-lpthread \
			-lstdc++

SRC = ./src
DESDIR = ./

GCC = /usr/bin/g++
CPPFLAGS = -g -finline-functions -Wall -Winline -pipe

TARGET1 = $(DESDIR)train
TARGET2 = $(DESDIR)train_debug
TARGET3 = $(DESDIR)test
TARGET4 = $(DESDIR)test_debug

OBJ1 = train_main.o gbdt.o common_func.o regression_tree.o gbdt_data.o
OBJ2 = train_main_debug.o gbdt_debug.o common_func_debug.o regression_tree_debug.o gbdt_data_debug.o
OBJ3 = test_main.o gbdt.o common_func.o regression_tree.o gbdt_data.o
OBJ4 = test_main_debug.o gbdt_debug.o common_func_debug.o regression_tree_debug.o gbdt_data_debug.o

all: clean $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)
	rm -rf output
	rm -f *.o
	mkdir output
	cp $(TARGET1) output
	cp $(TARGET2) output
	cp $(TARGET3) output
	cp $(TARGET4) output
        
$(TARGET1) : $(OBJ1)
	$(GCC) -g -o $@ $^ $(INCLUDEDIR) $(LIBDIR)
$(TARGET2) : $(OBJ2)
	$(GCC) -g -o $@ $^ $(INCLUDEDIR) $(LIBDIR)
$(TARGET3) : $(OBJ3)
	$(GCC) -g -o $@ $^ $(INCLUDEDIR) $(LIBDIR)
$(TARGET4) : $(OBJ4)
	$(GCC) -g -o $@ $^ $(INCLUDEDIR) $(LIBDIR)

%.o : $(SRC)/%.cpp
	$(GCC) $(CPPFLAGS) -DNDEBUG -c $< -o $@ $(INCLUDEDIR)
%_debug.o : $(SRC)/%.cpp
	$(GCC) $(CPPFLAGS) -c -DDEBUG $< -o $@ $(INCLUDEDIR)

clean:
	rm -rf *.o output log/* $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)
