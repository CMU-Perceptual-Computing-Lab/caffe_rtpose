#ifndef MODEL_DESCRIPTOR_FACTORY_H
#define MODEL_DESCRIPTOR_FACTORY_H

#include <memory>
#include "modelDescriptor.h"

/**
 *  A factory class. It allows the creation of the different kind of model descriptors (e.g. MPI_15, COCO_18).
 */
class ModelDescriptorFactory {
public:
    /** 
     * A class enum at which all the possible type of model descriptors are included.
     */
    enum class Type {
        MPI_15,
        COCO_18
    };

    /**
       * The only function of this factory method. It creates the desired kind of model descriptors (e.g. MPI_15, COCO_18).
       * @param type is a ModelDescriptorFactory::Type component, specifying the type of model descriptor.
       * @param modelDescriptorUniquePtr is a unique_ptr object with the created ModelDescriptor class.
       */
    const static void createModelDescriptor(const Type type, std::unique_ptr<ModelDescriptor> &modelDescriptorUniquePtr);
};

#endif // MODEL_DESCRIPTOR_FACTORY_H
