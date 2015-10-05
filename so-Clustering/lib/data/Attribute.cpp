
#include "data/Attribute.h"

using namespace std;
using namespace boost;


/**
 * @brief getIndex Get the index corresponding to the nominal attribute value
 * @param _attribute
 * @return
 */
int Attribute::getIndex(std::string _attribute) {
    int retIndex;
    // Insert the attribute in the attribute map and return the corresponding attribute index
    // If the attribute doesn't exist yet, create entry <attribute, index>
    unordered_map<string, int>::iterator mapIt = attributeMap.find(_attribute);
    if (mapIt == attributeMap.end()) {
        retIndex = index;
        attributeMap.insert(pair<string, int>(_attribute, retIndex));
        ++index;
    }
    else {
        // Else add exam to the student exams
        retIndex = mapIt->second;
    }
    return retIndex;
}


