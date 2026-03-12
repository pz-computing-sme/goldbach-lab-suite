/**
 * ============================================================================
 * RSST – Reduced Sum Set Test for Goldbach Conjecture
 * ============================================================================
 *
 * Author:   Vitor M. Pozza
 * Contact:  pozza.data.analyst@outlook.com
 * Date:     2025-02-17
 * Version:  2.1 (with progress display)
 *
 * Description:
 *   This program computes the Goldbach partition function G(n) for all even
 *   numbers up to a user‑defined limit. It generates and caches the list of
 *   primes, calculates cumulative sums, exports full data to CSV, and prints
 *   cumulative statistics for powers of ten intervals.
 *
 * Compilation (with OpenMP for parallelism):
 *   g++ -std=c++17 -O3 -march=native -fopenmp -o rsst rsst.cpp -lm
 *
 * Usage:
 *   ./rsst <limit>          (e.g., ./rsst 100000000 for 10⁸)
 *
 * Output:
 *   - primes_upto_<limit>.bin : Binary cache of primes (reused in future runs).
 *   - goldbach_full.csv       : Full data (n, G(n), S(n)).
 *   - Console: Cumulative statistics table (same as in the paper).
 *
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <cstdint>

using namespace std;

// ----------------------------------------------------------------------
// Type aliases for clarity
using u64 = uint64_t;
using u32 = uint32_t;

// ----------------------------------------------------------------------
// Structure to hold one Goldbach entry
struct GoldbachEntry {
    u64 n;           // even number
    u32 partitions;  // G(n)
    u64 cumulative;  // S(n) = sum_{m ≤ n} G(m)
};

// ----------------------------------------------------------------------
// Prime list management

/**
 * Saves a vector of primes to a binary file.
 */
void savePrimes(const vector<u32>& primes, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot create file " << filename << endl;
        return;
    }
    size_t size = primes.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(primes.data()), size * sizeof(u32));
    file.close();
    cout << "Primes saved to " << filename << " (" << size << " primes)" << endl;
}

/**
 * Loads a vector of primes from a binary file.
 * Returns true on success, false otherwise.
 */
bool loadPrimes(const string& filename, vector<u32>& primes) {
    ifstream file(filename, ios::binary);
    if (!file) return false;

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    primes.resize(size);
    file.read(reinterpret_cast<char*>(primes.data()), size * sizeof(u32));
    file.close();
    return true;
}

/**
 * Generates all primes up to 'limit' using the segmented sieve.
 * (Simple Eratosthenes is fine for limits up to ~10⁹ with enough memory.)
 */
vector<u32> generatePrimes(u64 limit) {
    cout << "Generating primes up to " << limit << "..." << endl;
    double t0 = omp_get_wtime();

    vector<bool> isPrime(limit + 1, true);
    isPrime[0] = isPrime[1] = false;

    u32 sqrtLimit = static_cast<u32>(sqrt(limit));
    for (u32 p = 2; p <= sqrtLimit; ++p) {
        if (isPrime[p]) {
            for (u64 m = static_cast<u64>(p) * p; m <= limit; m += p) {
                isPrime[m] = false;
            }
        }
    }

    vector<u32> primes;
    primes.reserve(static_cast<size_t>(limit / log(limit))); // approximate
    for (u32 i = 2; i <= limit; ++i) {
        if (isPrime[i]) primes.push_back(i);
    }

    double t1 = omp_get_wtime();
    cout << "Primes generated in " << (t1 - t0) << " s. Total: " << primes.size() << endl;
    return primes;
}

// ----------------------------------------------------------------------
// Core Goldbach computation

/**
 * Computes G(n) and S(n) for all even numbers 4 ≤ n ≤ limit.
 * Assumes primes list is already available.
 */
vector<GoldbachEntry> computeGoldbach(u64 limit, const vector<u32>& primes) {
    u64 numEvens = (limit - 4) / 2 + 1;
    vector<GoldbachEntry> results(numEvens);

    cout << "Computing Goldbach partitions for " << numEvens << " even numbers..." << endl;
    double t0 = omp_get_wtime();

    const u64 progressStep = max<u64>(1, numEvens / 100); // 1% steps

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1000)
        for (u64 i = 0; i < numEvens; ++i) {
            u64 n = 4 + 2 * i;
            u32 count = 0;
            u64 half = n / 2;

            // Use iterators for faster access
            auto itEnd = primes.end();
            for (auto it = primes.begin(); it != itEnd && *it <= half; ++it) {
                u32 p = *it;
                // Check if n-p is prime via binary search (primes are sorted)
                if (binary_search(primes.begin(), itEnd, n - p)) {
                    ++count;
                }
            }

            results[i].n = n;
            results[i].partitions = count;

            // Display progress every 1%
            if (i % progressStep == 0 && i > 0) {
                #pragma omp critical
                {
                    cout << "\rProgress: " << (100 * i / numEvens) << "%" << flush;
                }
            }
        }
    }

    cout << "\rProgress: 100%" << endl;

    // Compute cumulative sum S(n)
    u64 cumulative = 0;
    for (u64 i = 0; i < numEvens; ++i) {
        cumulative += results[i].partitions;
        results[i].cumulative = cumulative;
    }

    double t1 = omp_get_wtime();
    cout << "Computation finished in " << (t1 - t0) << " s." << endl;
    return results;
}

// ----------------------------------------------------------------------
// Output functions

/**
 * Saves full data (n, G(n), S(n)) to a CSV file.
 */
void saveToCSV(const vector<GoldbachEntry>& data, const string& filename) {
    cout << "Saving full data to " << filename << "..." << endl;
    ofstream file(filename);
    if (!file) {
        cerr << "Error: cannot create file " << filename << endl;
        return;
    }
    file << "n,G(n),S(n)\n";
    for (const auto& e : data) {
        file << e.n << "," << e.partitions << "," << e.cumulative << "\n";
    }
    file.close();
    cout << "Data saved." << endl;
}

/**
 * Prints cumulative statistics for powers of ten intervals (same as paper's Table 1).
 */
void printCumulativeStats(const vector<GoldbachEntry>& data, u64 limit) {
    // Determine powers of ten up to limit
    vector<u64> intervals;
    u64 pow10 = 10000; // start at 10⁴
    while (pow10 <= limit) {
        intervals.push_back(pow10);
        pow10 *= 10;
    }
    if (intervals.empty() || intervals.back() != limit) {
        intervals.push_back(limit);
    }

    size_t idx = 0;
    u64 count = 0;
    u32 minG = UINT32_MAX;
    u32 maxG = 0;
    u64 sumG = 0;
    double sumDens = 0.0;

    cout << fixed << setprecision(2);
    cout << "\n+----------------+----------+---------+---------+-----------+----------------+----------+\n";
    cout << "| Interval       | # Evens  | Min G   | Max G   | Mean G    | Mean Density   | Cp       |\n";
    cout << "+----------------+----------+---------+---------+-----------+----------------+----------+\n";

    for (u64 L : intervals) {
        while (idx < data.size() && data[idx].n <= L) {
            const auto& e = data[idx];
            ++count;
            if (e.partitions < minG) minG = e.partitions;
            if (e.partitions > maxG) maxG = e.partitions;
            sumG += e.partitions;
            sumDens += static_cast<double>(e.partitions) / e.n;
            ++idx;
        }

        double meanG = (count > 0) ? static_cast<double>(sumG) / count : 0.0;
        double meanDens = (count > 0) ? sumDens / count : 0.0;
        double lnL = log(static_cast<double>(L));
        double denom = L / (lnL * lnL);
        double Cp = (denom > 0) ? meanG / denom : 0.0;

        cout << "| 4 to " << setw(10) << L << " | "
             << setw(8) << count << " | "
             << setw(7) << minG << " | "
             << setw(7) << maxG << " | "
             << setw(9) << meanG << " | "
             << setw(14) << setprecision(6) << meanDens << " | "
             << setw(8) << setprecision(4) << Cp << " |\n";
    }
    cout << "+----------------+----------+---------+---------+-----------+----------------+----------+\n";
}

// ----------------------------------------------------------------------
// Main

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <limit>\n";
        return 1;
    }

    u64 limit = strtoull(argv[1], nullptr, 10);
    if (limit < 4) {
        cerr << "Limit must be at least 4.\n";
        return 1;
    }

    cout << "=============================================================\n";
    cout << "RSST – Goldbach Conjecture Data Generator (Optimized)\n";
    cout << "Limit: " << limit << " (" << (limit / 1'000'000) << " million)\n";
    cout << "=============================================================\n";

    double totalStart = omp_get_wtime();

    // --- Prime list management -----------------------------------------
    string primesFile = "primes_upto_" + to_string(limit) + ".bin";
    vector<u32> primes;

    if (!loadPrimes(primesFile, primes)) {
        cout << "Prime cache not found. Generating new list..." << endl;
        primes = generatePrimes(limit);
        savePrimes(primes, primesFile);
    } else {
        cout << "Loaded primes from cache: " << primesFile
             << " (" << primes.size() << " primes)" << endl;
    }

    // --- Core computation ---------------------------------------------
    auto data = computeGoldbach(limit, primes);

    // --- Save full data ------------------------------------------------
    saveToCSV(data, "goldbach_full.csv");

    // --- Print cumulative statistics -----------------------------------
    printCumulativeStats(data, limit);

    double totalEnd = omp_get_wtime();
    cout << "\nTotal execution time: " << (totalEnd - totalStart) << " s.\n";

    return 0;
}