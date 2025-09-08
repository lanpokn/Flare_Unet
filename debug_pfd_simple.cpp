/*
 * Simple debug version of PFD_WSL.cpp to understand what's happening
 */

#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_txt> <output_txt>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    std::ifstream event_file(input_path);
    if (!event_file.is_open()) {
        std::cerr << "Cannot open input file: " << input_path << std::endl;
        return 1;
    }
    
    int x, y, p;
    long double t, t0;
    int event_count = 0;
    
    // Read first event
    if (event_file >> x >> y >> p >> t0) {
        std::cout << "First event: x=" << x << ", y=" << y << ", p=" << p << ", t0=" << t0 << std::endl;
        event_count++;
        
        // Read remaining events
        while (event_file >> x >> y >> p >> t) {
            event_count++;
            long double timestamp = t - t0;
            
            if (event_count <= 5) {  // Show first 5 events
                std::cout << "Event " << event_count << ": x=" << x << ", y=" << y 
                         << ", p=" << p << ", t=" << t << ", timestamp=" << timestamp << std::endl;
            }
        }
    }
    
    event_file.close();
    
    std::cout << "Total events read: " << event_count << std::endl;
    
    // Write dummy output file
    std::ofstream out_file(output_path);
    out_file << "# Debug output - no actual processing" << std::endl;
    out_file.close();
    
    return 0;
}