#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/stl.h>
#include <vector>
#include <map>
namespace py = pybind11;
using namespace std;
 

typedef map<double, vector<double>> scene_class_set;


// copy from https://blog.csdn.net/tomwillow/article/details/81133932
template <typename T>
void combine_inner(T &data, int start, int n, int m, int depth, T temp,vector<T> &result)
{
    if (depth == m - 1)
    {
        //最内层循环 将temp加入result
        for (int i = start; i < n - (m - depth - 1); ++i)
        {
            temp[depth] = data[i];
            result.push_back(temp);
        }
    }
    else
        for (int i = start; i < n - (m - depth - 1);++i)
    {
        temp[depth] = data[i];//每层输出一个元素
        combine_inner(data,i + 1, n, m, depth+1,temp,result);
    }
}

//T可以调入vector<int>, string等，需要支持下标[]操作及size()函数
template <typename T>
vector<T> combine(T &data,int m)
{
    if (m <= 0)
        return{};
    int depth = 0;
    vector<T> result;
    T temp(m,0);
    combine_inner(data,0, data.size(), m, depth,temp,result);
    return result;
}



py::array_t<double> get_relation(
    py::array_t<double>& obj_class, // B x N
    py::array_t<int>& origin_relation, // B x N x N x num_relations
    py::array_t<double>& obj_distance, // B x N x N
    py::array_t<int>& context_size)// B
{

    auto object_class = obj_class.unchecked<2>();
    auto relation = origin_relation.unchecked<4>();
    auto distance = obj_distance.unchecked<3>();
    auto context = context_size.unchecked<1>();

    int bsz = obj_class.shape()[0];// batch size
    int num_object = obj_class.shape()[1];// objects number in a scene

    // prepare out relation
    py::array_t<int> out = py::array_t<int>(origin_relation.size());
    out.resize({origin_relation.shape()[0], origin_relation.shape()[1], origin_relation.shape()[2], origin_relation.shape()[3]});
    auto out_relation = out.mutable_unchecked<4>();

    for (int i = 0; i < origin_relation.shape()[0]; i++)
        {
            for (int j = 0; j < origin_relation.shape()[1]; j++)
            {
                for (int k = 0; k < origin_relation.shape()[2]; k++)
                {
                    for (int l = 0; l < origin_relation.shape()[3]; l++)
                    {
                        int value = relation(i, j, k, l);

                        out_relation(i, j, k, l) = value;
                    }
                
                }
            }
        }


    for (int i_bsz = 0; i_bsz < bsz; i_bsz++)
    {
        // get the class set in the scene 
        scene_class_set class_set;
        for (int i_obj = 0; i_obj < context(i_bsz); i_obj++)
        {
            if (isinf(object_class(i_bsz, i_obj)))//remove the padding class in the scene
            {
                continue;
            }

            if (class_set.find(object_class(i_bsz, i_obj)) != class_set.end())// is the object class in the set or not
            {
                vector<double> obj_index = class_set.find(object_class(i_bsz, i_obj)) -> second;  
                obj_index.push_back(i_obj);
                class_set[object_class(i_bsz, i_obj)] = obj_index;
            }
            else
            {
                vector<double> obj_index;
                obj_index.push_back(i_obj);
                class_set[object_class(i_bsz, i_obj)] = obj_index;

            }
            // 
        }

        //
        for (int tar_obj = 0; tar_obj < context(i_bsz); tar_obj++)
        {
            double tar_class = object_class(i_bsz, tar_obj);//get target object class
            // tranverse the class
            for (auto class_key = class_set.begin(); class_key != class_set.end(); ++class_key)
            {
                if (class_key->first == tar_class)
                {
                    continue;
                }
                vector<double> class_obj_index = class_key->second;//get object set belonging to target class
                double farthest_obj = class_obj_index[0];
                double closet_obj = class_obj_index[0];
                //find the closet and farthest
                for (int _class_obj_index = 0; _class_obj_index < class_obj_index.size(); _class_obj_index++)
                {
                    double temp_obj_index = class_obj_index[_class_obj_index];
                    if (distance(i_bsz, tar_obj, temp_obj_index) > distance(i_bsz, tar_obj, farthest_obj))
                    {
                        farthest_obj = temp_obj_index;
                    }

                    if (distance(i_bsz, tar_obj, temp_obj_index) < distance(i_bsz, tar_obj, closet_obj))
                    {
                        closet_obj = temp_obj_index;
                    }
                }
                out_relation(i_bsz, farthest_obj, tar_obj, 4) = 1;
                out_relation(i_bsz, closet_obj, tar_obj, 5) = 1;
            }
        }
    }


    return out;
}


py::array_t<double> get_relation_topn(
    py::array_t<double>& obj_class, // B x (N x n)
    py::array_t<double>& obj_prob, // B x (N x n)
    py::array_t<int>& origin_relation, // B x N x N x num_relations
    py::array_t<double>& obj_distance, // B x N x N
    py::array_t<int>& context_size,// B
    int n                           // the top n class for object
    )
{
    auto object_class = obj_class.unchecked<2>();
    auto object_prob = obj_prob.unchecked<2>();
    auto relation = origin_relation.unchecked<4>();
    auto distance = obj_distance.unchecked<3>();
    auto context = context_size.unchecked<1>();

    int bsz = origin_relation.shape()[0];// batch size
    int num_object = origin_relation.shape()[1];// objects number in a scene

    // prepare out relation
    py::array_t<int> out = py::array_t<int>(origin_relation.size());
    out.resize({origin_relation.shape()[0], origin_relation.shape()[1], origin_relation.shape()[2], origin_relation.shape()[3]});
    auto out_relation = out.mutable_unchecked<4>();

    for (int i = 0; i < origin_relation.shape()[0]; i++)
        {
            for (int j = 0; j < origin_relation.shape()[1]; j++)
            {
                for (int k = 0; k < origin_relation.shape()[2]; k++)
                {
                    for (int l = 0; l < origin_relation.shape()[3]; l++)
                    {
                        int value = relation(i, j, k, l);

                        out_relation(i, j, k, l) = value;
                    }
                
                }
            }
        }

    for (int i_bsz = 0; i_bsz < bsz; i_bsz++)
    {
        // get the class set in the scene 
        scene_class_set class_set;
        for (int i_obj = 0; i_obj < context(i_bsz) * n; i_obj++)
        {
            if (isinf(object_class(i_bsz, i_obj)))//remove the padding class in the scene
            {
                continue;
            }

            if (class_set.find(object_class(i_bsz, i_obj)) != class_set.end())// is the object class in the set or not
            {
                vector<double> obj_index = class_set.find(object_class(i_bsz, i_obj)) -> second;  
                obj_index.push_back(i_obj);
                class_set[object_class(i_bsz, i_obj)] = obj_index;
            }
            else
            {
                vector<double> obj_index;
                obj_index.push_back(i_obj);
                class_set[object_class(i_bsz, i_obj)] = obj_index;

            }
            // 
        }
        //
        for (int tar_obj = 0; tar_obj < context(i_bsz) * n; tar_obj++)
        {
            double tar_class = object_class(i_bsz, tar_obj);//get target object class
            // tranverse the class
            for (auto class_key = class_set.begin(); class_key != class_set.end(); ++class_key)
            {
                if (class_key->first == tar_class)
                {
                    continue;
                }
                vector<double> class_obj_index = class_key->second;//get object set belonging to target class
                int class_obj_num = class_obj_index.size();// numebr of object having the same label
                for(int _class_obj_num = 1; _class_obj_num <= class_obj_num; _class_obj_num++)
                {
                    // get index combinations of one class set in same size; 
                    auto combine_set = combine(class_obj_index, _class_obj_num);
                    double prob = 1;
                    for(int com_iter = 0; com_iter < combine_set.size(); com_iter++)
                    {
                        vector<double> _combine = combine_set[com_iter];// the index combinate of one class object
                        double farthest_obj = _combine[0];
                        double closet_obj = _combine[0];
                        for(int iter = 0; iter < _class_obj_num; iter++)//tranverse the whole set of a class
                        {   
                            double temp_index = class_obj_index[iter];
                            if(find(_combine.begin(), _combine.end(), temp_index) != _combine.end())
                            {
                                prob *= object_prob(i_bsz, temp_index);
                                if (distance(i_bsz, int(tar_obj/n), temp_index) > distance(i_bsz, int(tar_obj/n), farthest_obj))
                                {
                                    farthest_obj = temp_index;
                                }

                                if (distance(i_bsz, int(tar_obj/n), temp_index) < distance(i_bsz, int(tar_obj/n), closet_obj))
                                {
                                    closet_obj = temp_index;
                                }

                            }
                            else
                            {
                                prob *= (1 - object_prob(i_bsz, temp_index));
                            }
                        }
                        // vector<double> _combine = combine_set[com_iter];
                        // for (int _class_obj_index = 0; _class_obj_index < class_obj_num; _class_obj_index++)
                        // {
                        //     double farthest_obj = _combine[0];
                        //     double closet_obj = _combine[0];
                        //     //find the closet and farthest
                        //     double temp_obj_index = _combine[_class_obj_index];
                        //     if (distance(i_bsz, tar_obj, temp_obj_index) > distance(i_bsz, tar_obj, farthest_obj))
                        //     {
                        //         farthest_obj = temp_obj_index;
                        //     }

                        //     if (distance(i_bsz, tar_obj, temp_obj_index) < distance(i_bsz, tar_obj, closet_obj))
                        //     {
                        //         closet_obj = temp_obj_index;
                        //     }
                        // }
                        out_relation(i_bsz, int(farthest_obj/n), int(tar_obj/n), 4) += prob;
                        out_relation(i_bsz, int(closet_obj/n), int(tar_obj/n), 5) += prob;
                    }
                }
            }
        }
    }


    return out;
}




 
PYBIND11_MODULE(relation, m)
{
    // optional module docstring
    m.doc() = "pybind11 example plugin";
    // expose add function, and add keyword arguments and default arguments
    m.def("get_relation", &get_relation, "A function which cauculats the objects relations");
    m.def("get_relation_topn", &get_relation_topn,  "A function which cauculats the objects relations in Top N class");

}
 