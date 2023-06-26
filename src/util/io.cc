
#include "io.h"

namespace io {
// 通过时间戳判断是否为同时
bool IsSimu(timestamp_t t1, timestamp_t t2)//间隔为5000000
{
    if(std::abs(long(t1 - t2)) < 25000000)
        return true;
    return false;
}

// 在i——j之间，找到时间戳为离i.timestamp + t最近的迭代器
//template<typename T, typename T_iter>
//T_iter Move_Iterator(T_iter i, T_iter j, timestamp_t _t)// state表示返回的是否为同时
//{
//    timestamp_t t = i->timestamp + _t;
//    T_iter it = std::upper_bound(i, j, t,
//        [](const timestamp_t lhs, const T &rhs)
//        {return lhs < rhs.timestamp;});

//    if (it == i) return i;
//    else if (it == j) return j;
//    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
//    if ((it->timestamp - t) > (t - it_->timestamp)) return it_;
//    else return it;
//}

//template<typename T, typename T_iter>
//T_iter Move_Iterator(T_iter i, T_iter j, timestamp_t _t, bool &state)// state表示返回的是否为同时
//{
//    timestamp_t t = i->timestamp + _t;
//    T_iter it = std::upper_bound(i, j, t,
//        [](const timestamp_t lhs, const T &rhs)
//        {return lhs < rhs.timestamp;});

//    if (it == i)
//    {
//        state = IsSimu(i->timestamp, t);
//        return i;
//    }
//    else if (it == j)
//    {
//        state = IsSimu(j->timestamp, t);
//        return j;
//    }
//    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
//    if ((it->timestamp - t) > (t - it_->timestamp))// 这里是返回离t更近的时间戳对应的迭代器
//    {
//        state = IsSimu(it_->timestamp, t);
//        return it_;
//    }
//    else
//    {
//        state = IsSimu(it->timestamp, t);
//        return it;
//    }
//}

// 找到i——j之间，时间戳离t最近的迭代器

//template<typename T, typename T_iter>
//T_iter Find_Iterator(T_iter i, T_iter j, timestamp_t t)// state表示返回的是否为同时
//{
//    T_iter it = std::upper_bound(i, j, t,
//        [](const timestamp_t lhs, const T &rhs)
//        {return lhs < rhs.timestamp;});

//    if (it == i) return i;
//    else if (it == j) return j;
//    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
//    if ((it->timestamp - t) > (t - it_->timestamp)) return it_;
//    else return it;
//}

//template<typename T, typename T_iter>
//T_iter Find_Iterator(T_iter i, T_iter j, timestamp_t t, bool &state)// state表示返回的是否为同时
//{
//    T_iter it = std::upper_bound(i, j, t,
//        [](const timestamp_t lhs, const T &rhs)
//        {return lhs < rhs.timestamp;});

//    if (it == i)
//    {
//        state = IsSimu(i->timestamp, t);
//        return i;
//    }
//    else if (it == j)
//    {
//        state = IsSimu(j->timestamp, t);
//        return j;
//    }
//    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
//    if ((it->timestamp - t) > (t - it_->timestamp))// 这里是返回离t更近的时间戳对应的迭代器
//    {
//        state = IsSimu(it_->timestamp, t);
//        return it_;
//    }
//    else
//    {
//        state = IsSimu(it->timestamp, t);
//        return it;
//    }
//}

// ------------------------------------------------------------

} // namespace io
