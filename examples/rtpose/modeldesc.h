#ifndef MODELDESC_H__
#define MODELDESC_H__

#include <map>
#include <string>

struct ModelDescriptor {
  virtual ~ModelDescriptor() {}
  virtual const std::string& get_part_name(int n)=0;
  virtual int num_parts()=0;
  virtual int num_limb_seq()=0;
  virtual const int *get_limb_seq()=0;
  virtual const int *get_map_idx()=0;
  virtual const std::string name()=0;
};
struct MPIModelDescriptor : public ModelDescriptor {
	std::map<int, std::string> part2name;
  const int limbSeq[28] = {0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,14, 14,11, 11,12, 12,13, 14,8, 8,9, 9,10};
  const int mapIdx[28] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 43, 32, 33, 34, 35, 36, 37};
  virtual int num_parts() { return 15; }
  virtual int num_limb_seq() { return 14; }
  virtual const int *get_limb_seq() { return limbSeq; }
  virtual const int *get_map_idx() { return mapIdx; }
  virtual const std::string name() { return "MPI_15"; }

	MPIModelDescriptor() :
	part2name {
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
		{15, "Bkg"},
	} /* End initializers */	{
    for (int l=0;l<num_limb_seq();l++) {
      int la = limbSeq[2*l+0];
      int lb = limbSeq[2*l+1];
      int ma = mapIdx[2*l+0];
      int mb = mapIdx[2*l+1];
      part2name[ma] = part2name[la]+"->"+part2name[lb]+"(X)";
      part2name[mb] = part2name[la]+"->"+part2name[lb]+"(Y)";
    }
  }
  virtual const std::string& get_part_name(int n) {
    return part2name.at(n);
  }
};

struct COCOModelDescriptor : public ModelDescriptor {
	std::map<int, std::string> part2name;
  int limbSeq[38] = {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16, 0,15, 15,17, 2,16, 5,17};
	int mapIdx[38] = {31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46};
  virtual int num_parts() { return 18; }
  virtual int num_limb_seq() { return 38/2; }
  virtual const int *get_limb_seq() { return limbSeq; }
  virtual const int *get_map_idx() { return mapIdx; }
  virtual const std::string name() { return "COCO_18"; }

	COCOModelDescriptor() :
	part2name {
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
		{18, "Bkg"},
	} /* End initializers */	{
    for (int l=0;l<num_limb_seq();l++) {
      int la = limbSeq[2*l+0];
      int lb = limbSeq[2*l+1];
      int ma = mapIdx[2*l+0];
      int mb = mapIdx[2*l+1];
      part2name[ma] = part2name[la]+"->"+part2name[lb]+"(X)";
      part2name[mb] = part2name[la]+"->"+part2name[lb]+"(Y)";
    }

  }
  virtual const std::string& get_part_name(int n) {
    return part2name.at(n);
  }
};

#endif /* end of include guard:  */
