#ifndef TESTSETDESCRIPTION_H
#define TESTSETDESCRIPTION_H

#include <string>
#include <ostream>

/**
 * @brief The TestSetDescription class
 *
 * Description of clustering test sets
 */
class TestSetDescription {
public:
    TestSetDescription(std::string _name, std::string _description, std::string _dirName,
                       std::string _dataFilename, int _classIndex, int _numAttributes) :
        name(_name), description(_description), dirName(_dirName),
        dataFilename(_dataFilename), classIndex(_classIndex), numAttributes(_numAttributes) { }

    std::string getName() const { return name; }
    std::string getDescription() const { return description; }
    std::string getDirName() const { return dirName; }
    std::string getDataFilename() const { return dataFilename; }
    int getClassIndex() const { return classIndex; }
    int getNumAttributes() const { return numAttributes; }

    friend std::ostream& operator<<(std::ostream& os, const TestSetDescription& t);

private:
    //
    // TestSetDescription info: <name>, <description>, <directory name>, <data filename>
    // There two data files, namely <data filename>.data and <data names>.names
    //
    std::string name;
    std::string description;
    std::string dirName;
    std::string dataFilename;
    /**
     * @brief _classIndex Represents the index (1, ..., N) of the class attribute
     */
    int classIndex;
    /**
     * @brief numAttributes Number of attributes
     */
    int numAttributes;
};


#endif // TESTSETDESCRIPTION_H
