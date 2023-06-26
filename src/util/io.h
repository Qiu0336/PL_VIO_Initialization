// This file is part of The UMA-VI Dataset Tools
// Copyright (C) 2019-2021 David Zuñiga-Noël [dzuniga@uma.es]

#ifndef IO_H_
#define IO_H_

// STL
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/string.h"

namespace io {

static std::string DELIMITER = ",";
static unsigned int OUTPUT_PRECISION = 16;

// Types
using timestamp_t = std::uint64_t;

struct pose_t
{
    double tx, ty, tz, qw, qx, qy, qz;

    pose_t()
        : tx(0.0), ty(0.0), tz(0.0), qw(1.0), qx(0.0), qy(0.0), qz(0.0)
    { }

    pose_t(double tx, double ty, double tz, double qw, double qx, double qy, double qz)
        : tx(tx), ty(ty), tz(tz), qw(qw), qx(qx), qy(qy), qz(qz)
    {
        q_normalize();
    }

    // From Eigen Transform (constructor)
    template<typename Scalar, int Type>
    pose_t(const Eigen::Transform<Scalar, 3, Type> &T)
    {
        Eigen::Quaternion<Scalar> q(T.rotation());
        q.normalize();
        tx = T.translation()(0);
        ty = T.translation()(1);
        tz = T.translation()(2);
        qw = q.w();
        qx = q.x();
        qy = q.y();
        qz = q.z();
    }

    // To Eigen Transform (implicit conversion)
    template<typename Scalar, int Type>
    operator Eigen::Transform<Scalar, 3, Type>() const
    {
        Eigen::Quaternion<Scalar> q(qw, qx, qy, qz);
        q.normalize();

        Eigen::Transform<Scalar, 3, Type> T(q);

        T.translation()(0) = tx;
        T.translation()(1) = ty;
        T.translation()(2) = tz;

        return T;
    }

    // Inverse
    inline pose_t inverse()
    {
        return Eigen::Isometry3d(*this).inverse();
    }

    inline void q_normalize()
    {
        double norm = std::copysign(std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz), qw);
        qw /= norm;
        qx /= norm;
        qy /= norm;
        qz /= norm;
    }
};


struct trajectory_t
{
    timestamp_t timestamp;
    pose_t pose;

    trajectory_t()
        : timestamp(0), pose()
    { }

    trajectory_t(timestamp_t timestamp, const pose_t& pose)
        : timestamp(timestamp), pose(pose)
    { }
};

struct imu_data_t
{
    timestamp_t timestamp;
    double w_x, w_y, w_z, a_x, a_y, a_z;

    imu_data_t()
        : timestamp(0), w_x(0.0), w_y(0.0), w_z(0.0), a_x(0.0), a_y(0.0), a_z(0.0)
    { }

    imu_data_t(timestamp_t timestamp, double w_x, double w_y, double w_z, double a_x, double a_y, double a_z)
        : timestamp(timestamp), w_x(w_x), w_y(w_y), w_z(w_z), a_x(a_x), a_y(a_y), a_z(a_z)
    { }
};

struct state_t
{
    timestamp_t timestamp;
    pose_t pose;
    double v_x, v_y, v_z;
    double bw_x, bw_y, bw_z;
    double ba_x, ba_y, ba_z;

    state_t()
        : timestamp(0), pose(), v_x(0.0), v_y(0.0), v_z(0.0),
        bw_x(0.0), bw_y(0.0), bw_z(0.0), ba_x(0.0), ba_y(0.0), ba_z(0.0)
    { }

    state_t(timestamp_t timestamp, const pose_t& pose, double v_x, double v_y, double v_z,
            double bw_x, double bw_y, double bw_z, double ba_x, double ba_y, double ba_z)
        : timestamp(timestamp), pose(pose), v_x(v_x), v_y(v_y), v_z(v_z),
        bw_x(bw_x), bw_y(bw_y), bw_z(bw_z), ba_x(ba_x), ba_y(ba_y), ba_z(ba_z)
    { }
};


struct point_t
{
    int id;
    int cnt;// 被track的次数
    double x;
    double y;
    point_t(): id(0), cnt(0), x(0.0), y(0.0)
    { }

    point_t(int id, int cnt, double x, double y): id(id), cnt(cnt), x(x), y(y)
    { }

};

struct points_pf_t
{
    timestamp_t timestamp;
    std::vector<point_t> points;

    points_pf_t(timestamp_t timestamp, std::vector<point_t> points)
        : timestamp(timestamp), points(points)
    { }
};

struct line_t
{
    int id;// 类型？？
    int cnt;// 被track的次数
    double startX;
    double startY;
    double endX;
    double endY;

    line_t(): id(0), cnt(0), startX(0.0), startY(0.0), endX(0.0), endY(0.0)
    { }

    line_t(int id, int cnt, double startX, double startY, double endX, double endY)
        : id(id), cnt(cnt), startX(startX), startY(startY), endX(endX), endY(endY)
    { }
};

struct lines_pf_t
{
    timestamp_t timestamp;
    std::vector<line_t> lines;

    lines_pf_t(timestamp_t timestamp, std::vector<line_t> lines)
        : timestamp(timestamp), lines(lines)
    { }
};

using Trajectory = std::vector<trajectory_t>;
using ImuData = std::vector<imu_data_t>;
using State = std::vector<state_t>;
using Points = std::vector<point_t>;
using AllPoints = std::vector<points_pf_t>;
using Lines = std::vector<line_t>;
using AllLines = std::vector<lines_pf_t>;
// ------------------------------------------------------------

// Comparators

inline bool operator<(const trajectory_t& lhs, const trajectory_t& rhs) {
    return (lhs.timestamp < rhs.timestamp);
}

inline bool operator<(const imu_data_t& lhs, const imu_data_t& rhs) {
    return (lhs.timestamp < rhs.timestamp);
}

inline bool operator<(const state_t& lhs, const state_t& rhs) {
    return (lhs.timestamp < rhs.timestamp);
}

inline bool operator<(const point_t& lhs, const point_t& rhs) {
    return (lhs.id < rhs.id);
}

inline bool operator<(const line_t& lhs, const line_t& rhs) {
    return (lhs.id < rhs.id);
}
// ------------------------------------------------------------

// IO helpers

inline std::string to_string(int n, int w) {

    std::stringstream ss;
    ss << std::setw(w) << std::setfill('0') << n;// 控制间隔
    return ss.str();
}

inline std::string to_string(double n, int precision) {

    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << n;// std::fixed防止以科学计数法保存
    return ss.str();
}

// pose读
inline std::istream& operator>>(std::istream &lhs, pose_t &rhs)
{
    lhs >> rhs.tx >> rhs.ty >> rhs.tz >> rhs.qw >> rhs.qx >> rhs.qy >> rhs.qz;
//    lhs >> rhs.tx >> rhs.ty >> rhs.tz >> rhs.qx >> rhs.qy >> rhs.qz >> rhs.qw;
    rhs.q_normalize(); // to handle finite precision
    return lhs;
}

// pose写
inline std::ostream& operator<<(std::ostream& lhs, const pose_t& rhs)
{
    lhs << to_string(rhs.tx, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.ty, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.tz, OUTPUT_PRECISION) << DELIMITER
        << to_string(rhs.qw, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.qx, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.qy, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.qz, OUTPUT_PRECISION);
    return lhs;
}

// trajectory读
inline std::istream& operator>>(std::istream& lhs, trajectory_t& rhs)
{
    lhs >> rhs.timestamp >> rhs.pose;
    return lhs;
}

// trajectory写
inline std::ostream& operator<<(std::ostream& lhs, const trajectory_t& rhs)
{
    lhs << std::to_string(rhs.timestamp) << DELIMITER << rhs.pose;
    return lhs;
}

// imu读
inline std::istream& operator>>(std::istream& lhs, imu_data_t& rhs)
{
    lhs >> rhs.timestamp >> rhs.w_x >> rhs.w_y >> rhs.w_z >> rhs.a_x >> rhs.a_y >> rhs.a_z;
    return lhs;
}

// imu写
inline std::ostream& operator<<(std::ostream& lhs, const imu_data_t& rhs)
{
    lhs << rhs.timestamp << DELIMITER
        << to_string(rhs.w_x, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.w_y, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.w_z, OUTPUT_PRECISION) << DELIMITER
        << to_string(rhs.a_x, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.a_y, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.a_z, OUTPUT_PRECISION);
    return lhs;
}

// state读
inline std::istream& operator>>(std::istream &lhs, state_t &rhs) {

    lhs >> rhs.timestamp >> rhs.pose >> rhs.v_x >> rhs.v_y >> rhs.v_z
        >> rhs.bw_x >> rhs.bw_y >> rhs.bw_z >> rhs.ba_x >> rhs.ba_y >> rhs.ba_z;
    return lhs;
}

// state写
inline std::ostream& operator<<(std::ostream& lhs, const state_t& rhs) {

    lhs << rhs.timestamp << DELIMITER << rhs.pose
        << DELIMITER << to_string(rhs.v_x, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.v_y, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.v_z, OUTPUT_PRECISION)
        << DELIMITER << to_string(rhs.bw_x, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.bw_y, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.bw_z, OUTPUT_PRECISION)
        << DELIMITER << to_string(rhs.ba_x, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.ba_y, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.ba_z, OUTPUT_PRECISION);
    return lhs;
}

// point读
inline std::istream& operator>>(std::istream& lhs, point_t& rhs)
{
    lhs >> rhs.id >> rhs.cnt >> rhs.x >> rhs.y;
    return lhs;
}

// point写
inline std::ostream& operator<<(std::ostream& lhs, const point_t& rhs)
{
    lhs << rhs.id << DELIMITER << rhs.cnt << DELIMITER
        << to_string(rhs.x, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.y, OUTPUT_PRECISION);
    return lhs;
}

// line读
inline std::istream& operator>>(std::istream& lhs, line_t& rhs)
{
    lhs >> rhs.id >> rhs.cnt >> rhs.startX >> rhs.startY >> rhs.endX >> rhs.endY;
    return lhs;
}

// line写
inline std::ostream& operator<<(std::ostream& lhs, const line_t& rhs)
{
    lhs << rhs.id << DELIMITER << rhs.cnt << DELIMITER
        << to_string(rhs.startX, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.startY, OUTPUT_PRECISION) << DELIMITER
        << to_string(rhs.endX, OUTPUT_PRECISION) << DELIMITER << to_string(rhs.endY, OUTPUT_PRECISION);
    return lhs;
}

template<typename T>
inline std::vector<T> Read_File(const std::string &path)
{
    std::vector<T> records;
    std::ifstream input(path);
    if (!input.is_open()) return records;

    for (std::string line; std::getline(input, line);)//getline得到的字符串，一行中不同数据以","分开
    {
        if (line.empty() || line.front() == '#') continue;

        if (DELIMITER.compare(" ") != 0) line = StringReplace(line, DELIMITER, " ");
        // 把字符串line中所有的","都换成" "
        std::istringstream iss(line);
        T record;
        if (iss >> record) records.push_back(std::move(record));
    }

//    std::sort(records.begin(), records.end());// 按照时间戳排序
    return records;
}

template<typename T>
inline bool Write_File(const std::vector<T>& records, const std::string& path, const std::string& header = std::string())
{
    std::ofstream output(path);
    if (!output.is_open()) return false;

    if (!header.empty()) output << header << std::endl;
    for (const T& record : records)
        output << record << std::endl;

    output.close();
    return (!output.fail() && !output.bad());
}

// 通过时间戳判断是否为同时
bool IsSimu(timestamp_t t1, timestamp_t t2);//间隔为5000000

// 在i——j之间，找到时间戳为离i.timestamp + t最近的迭代器
template<typename T, typename T_iter>
T_iter Move_Iterator(T_iter i, T_iter j, timestamp_t _t)// state表示返回的是否为同时
{
    timestamp_t t = i->timestamp + _t;
    T_iter it = std::upper_bound(i, j, t,
        [](const timestamp_t lhs, const T &rhs)
        {return lhs < rhs.timestamp;});

    if (it == i) return i;
    else if (it == j) return j;
    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
    if ((it->timestamp - t) > (t - it_->timestamp)) return it_;
    else return it;
}

template<typename T, typename T_iter>
T_iter Move_Iterator(T_iter i, T_iter j, timestamp_t _t, bool &state)// state表示返回的是否为同时
{
    timestamp_t t = i->timestamp + _t;
    T_iter it = std::upper_bound(i, j, t,
        [](const timestamp_t lhs, const T &rhs)
        {return lhs < rhs.timestamp;});

    if (it == i)
    {
        state = IsSimu(i->timestamp, t);
        return i;
    }
    else if (it == j)
    {
        state = IsSimu(j->timestamp, t);
        return j;
    }
    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
    if ((it->timestamp - t) > (t - it_->timestamp))// 这里是返回离t更近的时间戳对应的迭代器
    {
        state = IsSimu(it_->timestamp, t);
        return it_;
    }
    else
    {
        state = IsSimu(it->timestamp, t);
        return it;
    }
}

// 找到i——j之间，时间戳离t最近的迭代器
template<typename T, typename T_iter>
T_iter Find_Iterator(T_iter i, T_iter j, timestamp_t t)// state表示返回的是否为同时
{
    T_iter it = std::upper_bound(i, j, t,
        [](const timestamp_t lhs, const T &rhs)
        {return lhs < rhs.timestamp;});

    if (it == i) return i;
    else if (it == j) return j;
    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
    if ((it->timestamp - t) > (t - it_->timestamp)) return it_;
    else return it;
}

template<typename T, typename T_iter>
T_iter Find_Iterator(T_iter i, T_iter j, timestamp_t t, bool &state)// state表示返回的是否为同时
{
    T_iter it = std::upper_bound(i, j, t,
        [](const timestamp_t lhs, const T &rhs)
        {return lhs < rhs.timestamp;});

    if (it == i)
    {
        state = IsSimu(i->timestamp, t);
        return i;
    }
    else if (it == j)
    {
        state = IsSimu(j->timestamp, t);
        return j;
    }
    T_iter it_ = std::prev(it);// it_为vector中it的上一个元素
    if ((it->timestamp - t) > (t - it_->timestamp))// 这里是返回离t更近的时间戳对应的迭代器
    {
        state = IsSimu(it_->timestamp, t);
        return it_;
    }
    else
    {
        state = IsSimu(it->timestamp, t);
        return it;
    }
}
// ------------------------------------------------------------

} // namespace io

#endif // IO_H_
