#ifndef MODEL_DESCRIPTOR_H
#define MODEL_DESCRIPTOR_H

#include <map>
#include <string>
#include <vector>

/**
 *  The ModelDescriptor class has all the information about the model descriptor (e.g. number of parts and their std::string names, limb sequence, etc.).
 */
class ModelDescriptor {
public:
    ModelDescriptor(const std::map<int, std::string> &partToNameBaseLine,
                    const std::vector<int> &limbSequenceInit,
                    const std::vector<int> &mapIdxInit);
    virtual ~ModelDescriptor();
    int get_number_parts();
    int number_limb_sequence();
    const std::vector<int> &get_limb_sequence();
    const std::vector<int> &get_map_idx();
    const std::string& get_part_name(int n);

private:
    const std::map<int, std::string> partToName;
    const std::vector<int> limbSequence;
    const std::vector<int> mapIdx;
    const unsigned long numberParts;
};

#endif // MODEL_DESCRIPTOR_H
