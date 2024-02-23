class PIDController:
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.Kp = 1900
        self.Ki = 2000
        self.Kd = 275
        self.bias = 0.0
        self.error_prev = 0.0
        self.error_sum = 0.0 
        self.dt_time = 1/60
        #to fix the integral windup problem
        self.count=0 
        return

    def reset(self):
        return

#TODO: Complete your PID control within this function. At the moment, it holds
#      only the bias. Your final solution must use the error between the 
#      target_pos and the ball position, plus the PID gains. You cannot
#      use the bias in your final answer. 
    def get_fan_rpm(self, vertical_ball_position):
        #having a max count to fix integral windup
        self.count += 1
        if self.count > 20:
            self.reset()
        error = self.target_pos - vertical_ball_position 
        self.error_sum=self.error_sum+error
        dt_error = (error - self.error_prev) 
        P = self.Kp*error
        I = self.Ki*self.error_sum*(self.dt_time) #multiplied by dt time to make it more sensitive
        D = self.Kd*dt_error/(self.dt_time) #dividing by dt time to make it more powerful
        output=P+I+D
        #we want to store our previous error for the next iteration
        self.error_prev = error
        return output
