#include <rclcpp/rclcpp.hpp>
#include "scale_adjuster/scale_adjuster.hpp"

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScaleAdjuster>());
  rclcpp::shutdown();
  return 0;
}