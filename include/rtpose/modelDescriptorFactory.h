#ifndef MODEL_DESCRIPTOR_FACTORY_H
#define MODEL_DESCRIPTOR_FACTORY_H

#include <memory>
#include "modelDescriptor.h"

/**
 *  ModelDescriptorFactory is a factory class. It allows the creation of the different kind of ModelDescriptor (e.g. MPI_15, COCO_18).
 */
class ModelDescriptorFactory {
public:
    /** 
     * A class enum at which all the possible type of ModelDescriptor are included. In order to add a new ModelDescriptor,
     * include its name in this enum and add a new 'else if' statement inside ModelDescriptorFactory::createModelDescriptor().
     */
    enum class Type {
        MPI_15,
        COCO_18,
    };

    /**
       * The only function of this factory method. It creates the desired kind of ModelDescriptor (e.g. MPI_15, COCO_18).
       * @param type is a const ModelDescriptorFactory::Type component, specifying the type of model descriptor.
       * @param modelDescriptorUniquePtr is a std::unique_ptr<ModelDescriptor> object with the new created ModelDescriptor class.
       */
    const static void createModelDescriptor(const Type type, std::unique_ptr<ModelDescriptor> &modelDescriptorUniquePtr);
};

#endif // MODEL_DESCRIPTOR_FACTORY_H
