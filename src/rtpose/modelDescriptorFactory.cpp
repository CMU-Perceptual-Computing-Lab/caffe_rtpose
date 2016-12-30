#include "rtpose/modelDescriptorFactory.h"
#include <stdexcept>

const void ModelDescriptorFactory::createModelDescriptor(const ModelDescriptorFactory::Type type, std::unique_ptr<ModelDescriptor> &modelDescriptorUniquePtr)
{
    if (type == Type::MPI_15)
    {
        modelDescriptorUniquePtr.reset(new ModelDescriptor{
            {{0,  "Head"},
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
             {15, "Bkg"}},                                                                                                      // partToName
            {0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,14, 14,11, 11,12, 12,13, 14,8, 8,9, 9,10},                                    // limbSequence
            {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 43, 32, 33, 34, 35, 36, 37}    // mapIdx
        });
    }

    else if (type == Type::COCO_18)
    {
        modelDescriptorUniquePtr.reset(new ModelDescriptor{
            {{0,  "Nose"},
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
             {18, "Bkg"}},                                                                                                                          // partToName
            { 1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17, 2,16,  5,17},   // limbSequence
            {31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46}   // mapIdx
        });
    }

    else
    {
        throw std::runtime_error{std::string{"Undefined ModelDescriptor selected."}};
    }
}
