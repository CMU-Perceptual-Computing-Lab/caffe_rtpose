#ifndef MODEL_DESCRIPTOR_H
#define MODEL_DESCRIPTOR_H

#include <exception>
#include <map>
#include <string>
#include <vector>

std::map<int, std::string> createPartToName(const std::map<int, std::string> &partToNameBaseLine, const std::vector<int> &limbSeq, const std::vector<int> &mapIdx)
{
    std::map<int, std::string> partToName = partToNameBaseLine;

    for (auto l=0;l<limbSeq.size() / 2;l++) {
        const auto la = limbSeq[2*l+0];
        const auto lb = limbSeq[2*l+1];
        const auto ma = mapIdx[2*l+0];
        const auto mb = mapIdx[2*l+1];
        partToName[ma] = partToName[la]+"->"+partToName[lb]+"(X)";
        partToName[mb] = partToName[la]+"->"+partToName[lb]+"(Y)";
    }

    return partToName;
}

struct ModelDescriptor {
    ModelDescriptor(const std::map<int, std::string> &partToNameBaseLine, const std::vector<int> &limbSeqInit, const std::vector<int> &mapIdxInit, const std::string &descriptorNameInit) :
        partToName{createPartToName(partToNameBaseLine, limbSeqInit, mapIdxInit)},
        limbSeq{limbSeqInit},
        mapIdx{mapIdxInit},
        numParts{partToNameBaseLine.size() - 1},
        descriptorName{descriptorNameInit}
    {
        if (limbSeqInit.size() != mapIdx.size())
            throw std::runtime_error{std::string{"limbSeqInit.size() should be equal to mapIdx.size()"}};
    }
    virtual ~ModelDescriptor() {}
    int num_parts() { return numParts; }
    int num_limb_seq() { return limbSeq.size() / 2; }
    const std::vector<int> &get_limb_seq() { return limbSeq; }
    const std::vector<int> &get_map_idx() { return mapIdx; }
    const std::string& get_part_name(int n) {
        return partToName.at(n);
    }
    const std::string name() { return descriptorName; }

private:
    const std::map<int, std::string> partToName;
    const std::vector<int> limbSeq;
    const std::vector<int> mapIdx;
    const unsigned long numParts;
    const std::string descriptorName;
};

struct MPIModelDescriptor : public ModelDescriptor {
    MPIModelDescriptor() :
    ModelDescriptor{
        std::map<int, std::string> {
            {0,  "Head"},
            {1,  "Neck"},
            {2,  "RShoulder"},
            {3,  "RElbow"},
            {4,  "RWrist"},
            {5,  "LShoulder"},
            {6,  "LElbow"},
            {7,  "LWrist"},
            {8,  "RHip"},
            {9,  "RKnee"},
            {10, "RAnkle"},
            {11, "LHip"},
            {12, "LKnee"},
            {13, "LAnkle"},
            {14, "Chest"},
            {15, "Bkg"}},
        {0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,14, 14,11, 11,12, 12,13, 14,8, 8,9, 9,10},   // limbSeq
        {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 43, 32, 33, 34, 35, 36, 37}, // mapIdx
        "MPI_15"} // model name
    {
    }
};

struct COCOModelDescriptor : public ModelDescriptor {
    COCOModelDescriptor() :
    ModelDescriptor {
        std::map<int, std::string> {
            {0,  "Nose"},
            {1,  "Neck"},
            {2,  "RShoulder"},
            {3,  "RElbow"},
            {4,  "RWrist"},
            {5,  "LShoulder"},
            {6,  "LElbow"},
            {7,  "LWrist"},
            {8,  "RHip"},
            {9,  "RKnee"},
            {10, "RAnkle"},
            {11, "LHip"},
            {12, "LKnee"},
            {13, "LAnkle"},
            {14, "REye"},
            {15, "LEye"},
            {16, "REar"},
            {17, "LEar"},
            {18, "Bkg"}},
        {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16, 0,15, 15,17, 2,16, 5,17},   // limbSeq
        {31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46}, // mapIdx
        "COCO_18"} // model name
    {
    }
};

#endif // MODEL_DESCRIPTOR_H
