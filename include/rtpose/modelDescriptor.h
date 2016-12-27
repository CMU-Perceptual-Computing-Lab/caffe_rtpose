#ifndef MODEL_DESCRIPTOR_H
#define MODEL_DESCRIPTOR_H

#include <map>
#include <string>
#include <vector>

/**
 *  The ModelDescriptor class has all the information about the Caffe model descriptor (e.g. number of parts and their std::string names, limb sequence, etc.).
 */
class ModelDescriptor {
public:
    /**
       * Constructor of the ModelDescriptor class.
       * @param partToNameBaseLine is a ModelDescriptorFactory::Type component, specifying the type of model descriptor.
       * @param limbSequence is a std::vector<int> with the new limb sequence.
       * @param mapIdx is a std::vector<int> with the same size as limbSequence with the mapping indexes.
       */
    ModelDescriptor(const std::map<int, std::string> &partToNameBaseLine,
                    const std::vector<int> &limbSequence,
                    const std::vector<int> &mapIdx);

    /**
       * Getter function which returns the number of parts of the model (e.g. 15 for MPI_15, 18 for COCO_18, etc.).
       * @return An int with a copy of mNumberParts.
       */
    int get_number_parts();

    /**
       * This function returns the number of limbs in the model.
       * @return An int with the number of limbs.
       */
    int number_limb_sequence();

    /**
       * Getter function which returns the limb sequence.
       * @return std::vector<int> reference of mLimbSequence.
       */
    const std::vector<int> &get_limb_sequence();

    /**
       * Getter function which returns the mapping indixes sequence.
       * @return std::vector<int> reference of mMapIdx.
       */
    const std::vector<int> &get_map_idx();

    /**
       * Mapping function which takes the limb or part index and returns its std::string name.
       * @param partIndex is a const int with the index of the desired part.
       * @return std::string with the name of the desired part.
       */
    const std::string &get_part_name(const int partIndex);

private:
    const std::map<int, std::string> mPartToName;
    const std::vector<int> mLimbSequence;
    const std::vector<int> mMapIdx;
    const int mNumberParts;
};

#endif // MODEL_DESCRIPTOR_H
