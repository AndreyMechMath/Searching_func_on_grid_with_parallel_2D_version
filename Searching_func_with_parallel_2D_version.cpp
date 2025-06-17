#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <unordered_map>

using namespace std;
using namespace std::chrono;

struct Point {
    int id;
    double x, y, z;
    double value;
};

struct Triangle {
    Point* a;
    Point* b;
    Point* c;
};

vector<Point> loadPoints(const string& filename) {
    ifstream file(filename);
    vector<Point> points;
    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        Point p;
        if (iss >> p.id >> p.x >> p.y >> p.z >> p.value) {
            points.push_back(p);
        }
    }

    unordered_map<int, Point*> pointMap;
    for (auto& p : points) {
        pointMap[p.id] = &p;
    }

    return points;
}

vector<Triangle> loadTriangles(const string& filename, vector<Point>& points) {
    ifstream file(filename);
    vector<Triangle> triangles;
    string line;
    bool inTriangleSection = false;
    unordered_map<int, Point*> pointMap;

    for (auto& p : points) {
        pointMap[p.id] = &p;
    }

    while (getline(file, line)) {
        if (line.empty()) continue;

        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (line.find("*Element") != string::npos ||
            line.find("*Element, type=type=STRI3") != string::npos) {
            inTriangleSection = true;
            continue;
        }

        if (inTriangleSection) {
            replace(line.begin(), line.end(), ',', ' ');
            istringstream iss(line);

            int id, id1, id2, id3;
            if (iss >> id >> id1 >> id2 >> id3) {
                if (pointMap.count(id1) && pointMap.count(id2) && pointMap.count(id3)) {
                    Triangle t;
                    t.a = pointMap[id1];
                    t.b = pointMap[id2];
                    t.c = pointMap[id3];
                    triangles.push_back(t);
                }
            }
        }
    }
    return triangles;
}

vector<Point> loadSearchPoints(const string& filename) {
    ifstream file(filename);
    vector<Point> points;
    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        Point p;
        if (iss >> p.x >> p.y) {
            p.z = 0;
            p.value = 0;
            points.push_back(p);
        }
    }
    return points;
}

class QuadTree {
private:
    struct QuadNode {
        double x_min, x_max, y_min, y_max;
        vector<Triangle*> triangles;
        unique_ptr<QuadNode> nw, ne, sw, se;

        QuadNode(double xmin, double xmax, double ymin, double ymax)
            : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}

        bool contains(const Triangle& t) const {
            double tri_min_x = min({ t.a->x, t.b->x, t.c->x });
            double tri_max_x = max({ t.a->x, t.b->x, t.c->x });
            double tri_min_y = min({ t.a->y, t.b->y, t.c->y });
            double tri_max_y = max({ t.a->y, t.b->y, t.c->y });

            return !(tri_max_x < x_min || tri_min_x > x_max ||
                tri_max_y < y_min || tri_min_y > y_max);
        }

        void split() {
            double x_mid = (x_min + x_max) / 2;
            double y_mid = (y_min + y_max) / 2;

            nw = make_unique<QuadNode>(x_min, x_mid, y_min, y_mid);
            ne = make_unique<QuadNode>(x_mid, x_max, y_min, y_mid);
            sw = make_unique<QuadNode>(x_min, x_mid, y_mid, y_max);
            se = make_unique<QuadNode>(x_mid, x_max, y_mid, y_max);

            for (auto tri : triangles) {
                if (nw->contains(*tri)) nw->triangles.push_back(tri);
                if (ne->contains(*tri)) ne->triangles.push_back(tri);
                if (sw->contains(*tri)) sw->triangles.push_back(tri);
                if (se->contains(*tri)) se->triangles.push_back(tri);
            }
            triangles.clear();
        }
    };

    unique_ptr<QuadNode> root;
    const size_t capacity;

public:
    QuadTree(double xmin, double xmax, double ymin, double ymax, size_t cap = 10)
        : root(make_unique<QuadNode>(xmin, xmax, ymin, ymax)), capacity(cap) {}

    void insert(Triangle* triangle) {
        insert(root.get(), triangle);
    }

    void insert(QuadNode* node, Triangle* triangle) {
        if (!node->contains(*triangle)) return;

        if (node->triangles.size() < capacity && !node->nw) {
            node->triangles.push_back(triangle);
        }
        else {
            if (!node->nw) node->split();
            insert(node->nw.get(), triangle);
            insert(node->ne.get(), triangle);
            insert(node->sw.get(), triangle);
            insert(node->se.get(), triangle);
        }
    }

    // Последовательная версия поиска
    double querySequential(double x, double y) const {
        return querySequential(root.get(), x, y);
    }

    double querySequential(const QuadNode* node, double x, double y) const {
        if (x < node->x_min || x > node->x_max || y < node->y_min || y > node->y_max) {
            return 0.0;
        }

        if (!node->nw) {
            for (auto tri : node->triangles) {
                double u, v, w;
                if (barycentric(*tri, x, y, u, v, w)) {
                    return u * tri->a->value + v * tri->b->value + w * tri->c->value;
                }
            }
            return 0.0;
        }

        if (auto res = querySequential(node->nw.get(), x, y); res != 0.0) return res;
        if (auto res = querySequential(node->ne.get(), x, y); res != 0.0) return res;
        if (auto res = querySequential(node->sw.get(), x, y); res != 0.0) return res;
        return querySequential(node->se.get(), x, y);
    }

    // Параллельная версия поиска (оставлена без изменений)
    double query(double x, double y) const {
        return query(root.get(), x, y);
    }

    double query(const QuadNode* node, double x, double y) const {
        if (x < node->x_min || x > node->x_max || y < node->y_min || y > node->y_max) {
            return 0.0;
        }

        if (!node->nw) {
            for (auto tri : node->triangles) {
                double u, v, w;
                if (barycentric(*tri, x, y, u, v, w)) {
                    return u * tri->a->value + v * tri->b->value + w * tri->c->value;
                }
            }
            return 0.0;
        }

        if (auto res = query(node->nw.get(), x, y); res != 0.0) return res;
        if (auto res = query(node->ne.get(), x, y); res != 0.0) return res;
        if (auto res = query(node->sw.get(), x, y); res != 0.0) return res;
        return query(node->se.get(), x, y);
    }

    static bool barycentric(const Triangle& t, double x, double y, double& u, double& v, double& w) {
        Point* a = t.a;
        Point* b = t.b;
        Point* c = t.c;

        double denom = (b->y - c->y) * (a->x - c->x) + (c->x - b->x) * (a->y - c->y);
        if (abs(denom) < 1e-10) return false;

        u = ((b->y - c->y) * (x - c->x) + (c->x - b->x) * (y - c->y)) / denom;
        v = ((c->y - a->y) * (x - c->x) + (a->x - c->x) * (y - c->y)) / denom;
        w = 1 - u - v;

        return u >= -1e-10 && v >= -1e-10 && w >= -1e-10;
    }
};

int main() {
    try {
        auto points = loadPoints("nodes.txt");
        if (points.empty()) throw runtime_error("No points loaded from nodes.txt");

        auto triangles = loadTriangles("triangulated_mesh.inp", points);
        if (triangles.empty()) throw runtime_error("No triangles loaded from triangulated_mesh.inp");

        auto search_points = loadSearchPoints("searching_points.txt");
        if (search_points.empty()) throw runtime_error("No search points loaded");

        double xmin = points[0].x, xmax = points[0].x;
        double ymin = points[0].y, ymax = points[0].y;
        for (const auto& p : points) {
            xmin = min(xmin, p.x);
            xmax = max(xmax, p.x);
            ymin = min(ymin, p.y);
            ymax = max(ymax, p.y);
        }

        double xmargin = (xmax - xmin) * 0.01;
        double ymargin = (ymax - ymin) * 0.01;
        xmin -= xmargin; xmax += xmargin;
        ymin -= ymargin; ymax += ymargin;

        auto start = high_resolution_clock::now();
        QuadTree qt(xmin, xmax, ymin, ymax, 10);
        for (auto& tri : triangles) {
            qt.insert(&tri);
        }
        auto build_time = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        cout << "QuadTree built in " << build_time.count() << " ms\n";

        // Последовательный поиск
        start = high_resolution_clock::now();
        vector<double> seq_results(search_points.size());
        for (int i = 0; i < search_points.size(); ++i) {
            seq_results[i] = qt.querySequential(search_points[i].x, search_points[i].y);
        }
        auto seq_time = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        cout << "Sequential search: " << seq_time.count() << " ms\n";

        // Параллельный поиск

        // Проверка активации OpenMP
#ifdef _OPENMP
        cout << "OpenMP enabled (Version: " << _OPENMP << ")" << endl;
#else
        cout << "WARNING: OpenMP disabled! Compile with -fopenmp" << endl;
#endif

        // Установка числа потоков
        const int PHYSICAL_CORES = 6;
        const int NUM_THREADS = PHYSICAL_CORES * 1.5;
        omp_set_num_threads(NUM_THREADS);

        // Проверка реального числа потоков
/*#pragma omp parallel
        {
#pragma omp single
            cout << "Actual threads: " << omp_get_num_threads() << endl;
        }
*/
        // Параллельный запуск
        start = high_resolution_clock::now();
        vector<double> par_results(search_points.size());
#pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0; i < search_points.size(); ++i) {
            par_results[i] = qt.query(search_points[i].x, search_points[i].y);
        }
        auto par_time = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        cout << "Parallel search: " << par_time.count() << " ms\n";
        cout << "Speedup: " << (double)seq_time.count() / par_time.count() << "x\n";

        cout << "Processed 1M points in " << par_time.count() << " ms" << endl;
        cout << "Throughput: " << 1e6 / par_time.count() << "K points/sec" << endl;
        // Сохранение результатов (используем параллельные результаты)
        ofstream out("output.txt");
        for (size_t i = 0; i < search_points.size(); ++i) {
            out << search_points[i].x << " " << search_points[i].y << " "
                << "0 " << par_results[i] << endl;
        }

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}