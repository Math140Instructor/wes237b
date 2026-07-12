CXX?=g++
CXXFLAGS=-Og -g

EXEC=huffman

INCLUDEDIR=include
SRCDIR=src
OBJDIR=objs

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS=$(patsubst $(SRCDIR)/*.cpp, $(OBJDIR)/*.o, $(SRCS))
INCLUDES=-I$(INCLUDEDIR)

all: $(SRCS) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS)-c $< -o $@

clean:
	rm -f $(EXEC)
	rm -f $(OBJDIR)/*.o
