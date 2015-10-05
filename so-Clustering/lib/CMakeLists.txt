#
# project
#
project(SOlib)

# Include "/lib" directory
include_directories(${PROJECT_SOURCE_DIR})
# Include ParadisEO directories
include_directories(${PROJECT_SOURCE_DIR}/../../ext/include/ParadisEO-2.0/eo/src/)
include_directories(${PROJECT_SOURCE_DIR}/../../ext/include/ParadisEO-2.0/mo/src/)
include_directories(${PROJECT_SOURCE_DIR}/../../ext/include/ParadisEO-2.0/moeo/src/)


add_subdirectory(algorithms)
add_subdirectory(chromosome)
add_subdirectory(data)
add_subdirectory(eval)
add_subdirectory(init)
add_subdirectory(testset)
add_subdirectory(utils)


#
# Set public header list (add your headers and source files here))
#
set(${PROJECT_NAME}_headers
        # algorithms/eo
        algorithms/eo/Crossover.h
        algorithms/eo/eoAlgoPointer.h
        algorithms/eo/eoCellularEA.h
        algorithms/eo/eoCellularEAMatrix.h
        algorithms/eo/eoCellularEARing.h
        algorithms/eo/eoDeterministicTournamentSelectorPointer.h
        algorithms/eo/eoGenerationContinuePopVector.h
        algorithms/eo/Mutation.h
        # chromosome
        chromosome/eoChromosome.h
        # data
        data/Attribute.h
        data/ProblemData.hpp
        # eval
        eval/eoClusteringEval.h
        eval/eoNumberEvalsCounter.h
        # eval/statistics
        eval/statistics/eoClusteringEvalWithStatistics.h
        # init
        init/eoClusteringInit.h
        # testset
        testset/TestSet.h
        testset/TestSetDescription.h
        testset/ClusteringTestSet.h
        # utils
        utils/CurrentDateTime.h
)


set(${PROJECT_NAME}_sources
        # chromosome
        chromosome/eoChromosome.cpp
        # data
        data/Attribute.cpp
        data/ProblemData.cpp
        # testset
        testset/TestSetDescription.cpp
        testset/ClusteringTestSet.cpp
        # utils
        utils/CurrentDateTime.cpp
        # lib
        MainApp.cpp
        MainAppClusteringDatasets.cpp
)



add_library(SOlib SHARED ${${PROJECT_NAME}_headers} ${${PROJECT_NAME}_sources})

# Put libraries in /usr/local/lib

# Include ParadisEO, Boost Regex, Armadillo, and libpca
target_link_libraries(${PROJECT_NAME} boost_regex eo es moeo cma eoutils ga armadillo pca)
#target_link_libraries(${PROJECT_NAME} boost_regex eo es moeo cma eoutils ga armadillo)

