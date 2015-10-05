#ifndef TESTSET_H
#define TESTSET_H

#include <iostream>
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "data/ProblemData.hpp"
#include "testset/TestSetDescription.h"


std::string read_from_file(char const* infile);



///////////////////////////////////////////////////////////////////////////////
//  Abstract class representing a test set
///////////////////////////////////////////////////////////////////////////////
class TestSet {

public:
//    std::string getName()          const { return name;        }
//    std::string getDescription()   const { return description; }
    TestSetDescription const &getDescription() const { return description; }
    std::string getRootDirectory() const { return rootDir;     }
    // Get timetable problem data
//    boost::shared_ptr<ProblemData> const& getProblemData() const {
//        return problemData;
//    }
    ProblemData const &getProblemData() const {
        return *problemData.get();
    }
    ProblemData &getProblemData() {
        return *problemData.get();
    }


    friend std::ostream& operator<<(std::ostream& _os, const TestSet& _t);

protected:
    // Protected constructor
//    TestSet(std::string _name, std::string _description, std::string _rootDir,
//            boost::shared_ptr<ProblemData> _problemData)
//        : name(_name), description(_description), rootDir(_rootDir), problemData(_problemData)
//    { }

    TestSet(TestSetDescription const &_description, std::string _rootDir,
            boost::shared_ptr<ProblemData> _problemData)
        : description(_description), rootDir(_rootDir), problemData(_problemData)
    { }


    // Pure virtual method
    virtual void load() = 0;

    // Set timetable problem data
    void setTimetableProblemData(boost::shared_ptr<ProblemData> const&  _timetableProblemData) {
        problemData = _timetableProblemData;
    }

private:
//    std::string name;
//    std::string description;
    TestSetDescription const &description;
    std::string rootDir;

protected:
    // Structure containing the problem data
    boost::shared_ptr<ProblemData> problemData;
};



#endif // TESTSET_H
