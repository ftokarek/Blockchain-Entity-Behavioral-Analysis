#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

int main() 
{
    std::ifstream infile("../../data/btc_and_eth_data.csv");
    std::ofstream btcfile("../../data/btc_data.csv");
    std::ofstream ethfile("../../data/eth_data.csv");

    if (!infile.is_open() || !btcfile.is_open() || !ethfile.is_open()) 
    {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(infile, line); // header
    btcfile << line << "\n";
    ethfile << line << "\n";

    while (std::getline(infile, line)) 
    {
        std::stringstream ss(line);
        std::string field;
        std::string currency;

        // Find value in the Currency column (index 8, zero-based)
        for (int i = 0; i <= 8; ++i) 
        {
            if (!std::getline(ss, field, ',')) 
                break;
            if (i == 8) 
                currency = field;
        }

        if (currency == "BTC") 
        {
            btcfile << line << "\n";
        } 
        else if (currency == "ETH") 
        {
            ethfile << line << "\n";
        }
    }

    infile.close();
    btcfile.close();
    ethfile.close();
    std::cout << "Splitting finished." << std::endl;
    return 0;
}