#ifndef CAFFE_CPM_UTIL_DB_HPP
#define CAFFE_CPM_UTIL_DB_HPP

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

DB* GetDB(CPMDataParameter::DB backend);
DB* GetDB(CocoDataParameter::DB backend);
DB* GetDB(CPMBottomUpDataParameter::DB backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_CPM_UTIL_DB_HPP
