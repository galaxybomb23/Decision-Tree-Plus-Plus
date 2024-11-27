#include <gtest/gtest.h>

#include "Utility.hpp"

class UtilityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // called before each test
    }

    void TearDown() override
    {
        // called after each test ends
    }

    // Class members to be used in tests
};

TEST_F(UtilityTest, sampleIndex)
{
    // Test the sampleIndex function on floating point numbers
    vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
    pair<double, size_t> result = minimum(vec);
    ASSERT_EQ(result.first, 1.0);
    ASSERT_EQ(result.second, 0);

    // Test the sampleIndex function on integers
    vector<int> vec2 = {23, 41, 12345, 123, 5, 51, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    pair<int, size_t> result2 = minimum(vec2);
    ASSERT_EQ(result2.first, 2);
    ASSERT_EQ(result2.second, 6);

    // Test the sampleIndex function on strings
    vector<string> vec3 = {"hello", "world", "this", "is", "a", "test"};
    pair<string, size_t> result3 = minimum(vec3);
    ASSERT_EQ(result3.first, "a");
    ASSERT_EQ(result3.second, 4);
}

TEST_F(UtilityTest, convert)
{
    // Test the convert function on a string
    string str = "3.14159";
    double result = convert(str);
    ASSERT_EQ(result, 3.14159);

    // Test the convert function on a string
    string str2 = "2.71828";
    double result2 = convert(str2);
    ASSERT_EQ(result2, 2.71828);

    // Test the convert function on a integer
    string str3 = "42";
    double result3 = convert(str3);
    ASSERT_EQ(result3, 42.0);
}

TEST_F(UtilityTest, ClosestSamplingPoint)
{
    // Test the ClosestSamplingPoint function on a vector of integers
    vector<double> vec = {1, 2, 3, 5, 8, 13, 21};
    optional<double> result = ClosestSamplingPoint(vec, 20);
    ASSERT_EQ(result.value(), 21);

    // Test the ClosestSamplingPoint function on a vector of doubles
    vector<double> vec2 = {1.0, 2.0, 3.0, 4.0, 5.0};
    optional<double> result2 = ClosestSamplingPoint(vec2, 3.7);
    ASSERT_EQ(result2.value(), 4.0);
}

// test the cleanup CSV function
TEST_F(UtilityTest, cleanupCSV)
{
    // Test the CleanupCsvString function on a string
    vector<tuple<string, string>> tests = {
        {"hello, world, this, is, a, test", "hello,world,this,is,a,test"},
        {"hello, , world, , test", "hello,NA,world,NA,test"},
        {"hello, \"quoted, text\", world", "hello,quoted_text,world"},
        {"hello, world, this, is, a, test, ", "hello,world,this,is,a,test,NA"},
        {"hello, world, this, is, a, test, ,", "hello,world,this,is,a,test,NA,NA"}};

    for (auto const &test : tests)
    {
        string result = CleanupCsvString(std::get<0>(test));
        ASSERT_EQ(result, std::get<1>(test));
    }
}