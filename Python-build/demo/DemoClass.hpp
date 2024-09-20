#ifndef PYTHON_DEMO_HPP
#define PYTHON_DEMO_HPP

// Include necessary headers
#include <string>
#include <iostream>
#include <map>
class DemoClass
{
public:
    DemoClass(int a, int b);
    ~DemoClass();

    void print();
    int add();
    int multiply();
    int subtract();
    float divide();
    int getA();
    int getB();
    void setA(int a);
    void setB(int b);
    bool isEven();
    void calcFibAB();
    std::map<std::string, int> getMap();
    void doughnut(int fps, int distance, float increment, int refreshRate, int xpos, int ypos);

private:
    int a;
    int b;
};
#endif // PYTHON_DEMO_HPP