
#include "data/ProblemData.hpp"


using namespace std;


ostream& operator<<(ostream& _os, const ProblemData& _problemData) {

    _os << endl << "ProblemData Info:" << endl
        << "NumSamples: " << _problemData.getNumSamples() << endl
        << "NumFeaturesPerSample: " << _problemData.getNumFeaturesPerSample() << endl;

//    _os <<  "Samples and classes:" << endl;

//    // Get samples
//    auto const &samples = _problemData.getSamples();
//    // Get class labels
//    auto const &classLabels = _problemData.getClassLabels();
//    // Get sample classes
//    auto const &samplesClasses = _problemData.getSamplesClasses();
//    for (int i = 0; i < samples.size(); ++i) {
//        _os << "Sample # " << (i+1) << ": ";
//        for (int j = 0; j < _problemData.getNumFeaturesPerSample(); ++j) {
//            _os << samples[i][j] << " ";
//        }
//        _os << " - class label: " << classLabels[samplesClasses[i]] << ", class index: " << samplesClasses[i];
//        _os << endl;
//    }
//    cout << "[ProblemData::operator<<] Attribute info:"  << endl;
//    cout << "atts.size() = " << _problemData.getAttributes().size() << endl;
//    for (auto & att : _problemData.getAttributes())
//        cout << "att.min() = " << att.min() << ", att.max() = " << att.max() << endl;
//    cin.get();

    return _os;
}


