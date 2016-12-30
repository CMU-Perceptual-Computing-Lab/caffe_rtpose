#include "rtpose/modelDescriptor.h"
#include <stdexcept>

std::map<int, std::string> createPartToName(const std::map<int, std::string> &partToNameBaseLine,
                                            const std::vector<int> &limbSequence,
                                            const std::vector<int> &mapIdx)
{
    std::map<int, std::string> partToName = partToNameBaseLine;

    for (auto l=0;l<limbSequence.size() / 2;l++) {
        const auto la = limbSequence[2*l+0];
        const auto lb = limbSequence[2*l+1];
        const auto ma = mapIdx[2*l+0];
        const auto mb = mapIdx[2*l+1];
        partToName[ma] = partToName[la]+"->"+partToName[lb]+"(X)";
        partToName[mb] = partToName[la]+"->"+partToName[lb]+"(Y)";
    }

    return partToName;
}

ModelDescriptor::ModelDescriptor(const std::map<int, std::string> &partToNameBaseLine,
                                 const std::vector<int> &limbSequence,
                                 const std::vector<int> &mapIdx) :
    mPartToName{createPartToName(partToNameBaseLine, limbSequence, mapIdx)},
    mLimbSequence{limbSequence},
    mMapIdx{mapIdx},
    mNumberParts{(int)partToNameBaseLine.size() - 1}
{
    if (limbSequence.size() != mMapIdx.size())
        throw std::runtime_error{std::string{"limbSequence.size() should be equal to mMapIdx.size()"}};
}

int ModelDescriptor::get_number_parts() {
    return mNumberParts;
}

int ModelDescriptor::number_limb_sequence() {
    return mLimbSequence.size() / 2;
}

const std::vector<int> &ModelDescriptor::get_limb_sequence() {
    return mLimbSequence;
}

const std::vector<int> &ModelDescriptor::get_map_idx() {
    return mMapIdx;
}

const std::string &ModelDescriptor::get_part_name(const int partIndex) {
    return mPartToName.at(partIndex);
}
