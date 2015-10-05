#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H


#include <string>
#include <boost/unordered_map.hpp>
#include <limits>


/**
 * @brief The Attribute class - Class to represent an attribute
 *
 * Attributes can be of one of the types: numeric, nominal, string, or date
 *
 */
class Attribute {

public:
    /**
     * @enum AttrValueEnum
     * @brief Types of data
     */
    enum AttrValueEnum {
        /** date */
        DATE,
        /** string */
        STRING,
        /** numeric */
        NUMERIC,
        /** nominal */
        NOMINAL,
        /** unknown */
        UNKNOWN_VAL
    };

    /**
     * @brief Attribute Default constructor
     *
     *   Attribute integer mapping starts at 1
     */
    Attribute() : index(1), minValue(std::numeric_limits<double>::max()),
        maxValue(std::numeric_limits<double>::min()) { }

    /**
     * @brief Name of this attribute
     * @return name
     */
    std::string getName() const;
    /**
     * @brief setName
     * @param _name
     */
    void setName(std::string _name);

    /**
     * @brief Type of this attribute
     * @return type
     */
    AttrValueEnum getType() const;
    /**
     * @brief setType
     * @param _type
     */
    void setType(AttrValueEnum const &_type);

    /**
     * @brief getIndex Get the index corresponding to the nominal attribute value
     * @param _attribute
     * @return
     */
    int getIndex(std::string _attribute);

    /**
     * @brief getNumAttributes
     * @return
     */
    int getNumAttributes() const;
    /**
     * @brief setNumAttributes
     * @param value
     */
    void setNumAttributes(int value);
    /**
     * @brief updateMinMax
     * @param feature
     */
    void updateMinMax(double feature);
    /**
     * @brief min
     * @return
     */
    double min() const;
    /**
     * @brief max
     * @return
     */
    double max() const;


private:
    /**
     * attribute name
     */
    std::string name;
    /**
     * attribute type
     */
    AttrValueEnum type;
    /**
     * @brief attributeMap
     */
    boost::unordered_map<std::string, int> attributeMap;
    /**
     * @brief index Attribute index
     */
    int index;
    /**
     * @brief numAttributes Number of attributes
     */
    int numAttributes;
    /**
     * @brief minValue
     */
    double minValue;
    /**
     * @brief maxValue
     */
    double maxValue;
};


/**
 * @brief Attribute::name
 * @return
 */
inline std::string Attribute::getName() const {
    return name;
}
/**
 * @brief setName
 * @param _name
 */
inline void Attribute::setName(std::string _name) {
    name = _name;
}

/**
 * @brief Attribute::type
 * @return
 */
inline Attribute::AttrValueEnum Attribute::getType() const {
    return type;
}
/**
 * @brief setType
 * @param _type
 */
inline void Attribute::setType(AttrValueEnum const &_type) {
    type = _type;
}
/**
 * @brief Attribute::getNumAttributes
 * @return
 */
inline int Attribute::getNumAttributes() const
{
    return numAttributes;
}
/**
 * @brief Attribute::setNumAttributes
 * @param value
 */
inline void Attribute::setNumAttributes(int value)
{
    numAttributes = value;
}

/**
 * @brief updateMinMax
 * @param feature
 */
inline void Attribute::updateMinMax(double feature) {
    // Update min value
    if (feature < minValue)
        minValue = feature;
    // Update max value
    if (feature > maxValue)
        maxValue = feature;
}

/**
 * @brief min
 * @return
 */
inline double Attribute::min() const {
    return minValue;
}

/**
 * @brief max
 * @return
 */
inline double Attribute::max() const {
    return maxValue;
}



#endif // ATTRIBUTE_H









