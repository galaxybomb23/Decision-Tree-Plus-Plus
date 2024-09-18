#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <cmath>
#include "DemoClass.hpp"

DemoClass::DemoClass(int a, int b) : a(a), b(b) {}
DemoClass::~DemoClass() {}

void DemoClass::print()
{
    std::cout << "a: " << a << " b: " << b << std::endl;
}

int DemoClass::add()
{
    return a + b;
}

int DemoClass::multiply()
{
    return a * b;
}

int DemoClass::subtract()
{
    return a - b;
}

float DemoClass::divide()
{
    return static_cast<float>(a) / b;
}

int DemoClass::getA()
{
    return a;
}

int DemoClass::getB()
{
    return b;
}

void DemoClass::setA(int a)
{
    this->a = a;
}

void DemoClass::setB(int b)
{
    this->b = b;
}

bool DemoClass::isEven()
{
    return this->add() % 2 == 0;
}

void DemoClass::calcFibAB()
{
    int current = 0;
    int next = 1;
    while (current <= b)
    {
        if (current >= a)
        {
            std::cout << current << " ";
        }
        int temp = current;
        current = next;
        next = temp + next;
    }
    std::cout << std::endl;
}

std::map<std::string, int> DemoClass::getMap()
{
    std::map<std::string, int> m;
    m["a"] = a;
    m["b"] = b;
    return m;
}

void DemoClass::doughnut(int fps, int distance, float increment, int refreshRate, int xpos, int ypos)
{
    int k;
    float A = 0, B = 0;
    float z[1760];
    char b[1760];
    float counter = .01;

    std::cout << "\x1b[2J"; // Clear screen

    while (true)
    {

        // sleep to meet the desired FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
        memset(b, 32, 1760); // Initialize buffer with spaces
        memset(z, 0, 7040);  // Initialize z-buffer with zeroes

        for (float j = 0; j < 6.28; j += 0.17)
        {
            for (float i = 0; i < 6.28; i += 0.02)
            {
                float c = std::sin(i);
                float d = std::cos(j);
                float e = std::sin(A);
                float f = std::sin(j);
                float g = std::cos(A);
                float h = d + counter;
                float D = 1 / (c * h * e + f * g + 5);
                float l = std::cos(i);
                float m = std::cos(B);
                float n = std::sin(B);
                float t = c * h * g - f * e;

                int x = xpos + 30 * D * (l * h * m - t * n);
                int y = ypos + 15 * D * (l * h * n + t * m);
                int o = x + 80 * y;
                int N = 8 * ((f * e - c * d * g) * m - c * d * e - f * g - l * d * n);

                if (y > 0 && y < 22 && x > 0 && x < 80 && D > z[o])
                {
                    z[o] = D;
                    b[o] = ".,-~:;=!*#$@"[N > 0 ? N : 0];
                }
            }
        }

        std::cout << "\x1b[H"; // Move cursor to top left
        for (k = 0; k < 1760; k++)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(refreshRate));
            std::cout << (k % 80 ? b[k] : '\n');
        }

        A += 0.04;
        B += 0.02;

        // make counter oscillate between 0 and distance
        if (counter >= distance || counter <= 0)
        {
            increment *= -1;
        }
        counter += increment;
    }
}