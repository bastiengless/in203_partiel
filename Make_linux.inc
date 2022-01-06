CC = mpicc
CXX = mpic++
LIB = -lmpi
MPIOPT = -oversubscribe
CXXFLAGS+=$(mpic++ --showme:compile)
LIBS+=$(mpic++ --showme:link)

