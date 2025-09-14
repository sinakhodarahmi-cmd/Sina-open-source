# Sina-open-source
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

// Simple Logistic Regression model
struct MoodModel {
    vector<double> w; // weights
    double bias;
    double lr; // learning rate

    MoodModel(int n_features, double lr_=0.05) {
        w.resize(n_features, 0.0);
        bias = 0.0;
        lr = lr_;
    }

    // Sigmoid activation
    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Prediction: returns probability of "good mood"
    double predict(const vector<double>& x) {
        double s = bias;
        for (size_t i = 0; i < w.size(); i++)
            s += w[i] * x[i];
        return sigmoid(s);
    }

    // Update weights using one sample (online learning)
    void update(const vector<double>& x, int y) {
        double pred = predict(x);
        double error = y - pred;
        for (size_t i = 0; i < w.size(); i++)
            w[i] += lr * error * x[i];
        bias += lr * error;
    }
};

int main() {
    cout << "=== Mood Predictor (tiny AI in C++) ===\n";
    cout << "Features: SleepHours, StudyHours, CoffeeCups, StressLevel\n";
    cout << "Target: Mood (1 = good, 0 = bad)\n\n";

    MoodModel model(4, 0.1);

    // Simple training dataset (you can expand it)
    vector<vector<double>> X = {
        {8, 2, 1, 2},   // slept well, studied a bit, low stress -> good mood
        {4, 8, 5, 9},   // little sleep, too much stress -> bad mood
        {7, 3, 2, 4},   // balanced -> good mood
        {3, 10, 6, 8}   // exhausted -> bad mood
    };
    vector<int> Y = {1, 0, 1, 0};

    // Train for some epochs
    for (int epoch = 0; epoch < 2000; epoch++) {
        for (size_t i = 0; i < X.size(); i++)
            model.update(X[i], Y[i]);
    }

    cout << "Training complete!\n";
    cout << "Now enter your own data (Sleep Study Coffee Stress):\n";

    double a,b,c,d;
    while (cin >> a >> b >> c >> d) {
        vector<double> sample = {a, b, c, d};
        double prob = model.predict(sample);
        int mood = (prob >= 0.5 ? 1 : 0);

        cout << fixed << setprecision(3);
        if (mood == 1)
            cout << "Predicted Mood: GOOD (prob = " << prob << ")\n";
        else
            cout << "Predicted Mood: BAD (prob = " << prob << ")\n";
    }

    return 0;
}
