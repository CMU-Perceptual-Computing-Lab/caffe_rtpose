#include "rtpose/modelDescriptor.h"
#include <exception>

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
                                 const std::vector<int> &limbSequenceInit,
                                 const std::vector<int> &mapIdxInit) :
    partToName{createPartToName(partToNameBaseLine, limbSequenceInit, mapIdxInit)},
    limbSequence{limbSequenceInit},
    mapIdx{mapIdxInit},
    numberParts{partToNameBaseLine.size() - 1}
{
    if (limbSequenceInit.size() != mapIdx.size())
        throw std::runtime_error{std::string{"limbSequenceInit.size() should be equal to mapIdx.size()"}};
}

ModelDescriptor::~ModelDescriptor() {}

int ModelDescriptor::get_number_parts() {
    return numberParts;
}

int ModelDescriptor::number_limb_sequence() {
    return limbSequence.size() / 2;
}

const std::vector<int> &ModelDescriptor::get_limb_sequence() {
    return limbSequence;
}

const std::vector<int> &ModelDescriptor::get_map_idx() {
    return mapIdx;
}

const std::string& ModelDescriptor::get_part_name(int n) {
    return partToName.at(n);
}
