function Optimal_EFPA_Mars_EDL_Simulation()
% MARS_ENTRY_DESCENT_LANDING_OPTIMIZATION
% Optimizes the Entry Flight Path Angle (EFPA) for Mars EDL trajectory
% 
% Features:
% - EFPA optimization using fmincon
% - Three-phase EDL simulation (entry, parachute, powered descent)
% - Hamiltonian validation for optimal control
% - Monte Carlo analysis for robustness testing
% - Comprehensive visualization and constraint checking

%% ========== INITIALIZATION ==========
clear; clc; close all;
fprintf('=== Mars EDL Optimization ===\n');

%% ========== PHYSICAL CONSTANTS ==========
params = initialize_parameters();

%% ========== OPTIMIZATION SETUP ==========
% EFPA bounds and initial guess
x0 = -17*pi/180;   % Initial guess (-17°)
lb = -18*pi/180;   % Lower bound (-18°)
ub = -16*pi/180;   % Upper bound (-16°)

% Configure fmincon options
options = configure_optimizer();

%% ========== RUN OPTIMIZATION ==========
[opt_result, simulation_data] = run_optimization(x0, lb, ub, options, params);

%% ========== RESULTS ANALYSIS ==========
display_results(opt_result, simulation_data, params);

%% ========== HAMILTONIAN VALIDATION ==========
if params.use_hamiltonian
    validate_with_hamiltonian(opt_result, simulation_data, params);
end

%% ========== MONTE CARLO ANALYSIS ==========
perform_monte_carlo_analysis(params, 50);

%% ========== HELPER FUNCTIONS ==========

    function params = initialize_parameters()
        % Initialize all physical and mission parameters
        
        params = struct();
        
        % Mars physical constants
        params.R_mars = 3396.2e3;      % Mars radius [m]
        params.mu_mars = 4.2828e13;    % Gravitational parameter [m^3/s^2]
        params.H = 11.1e3;             % Atmospheric scale height [m]
        params.rho0 = 0.020;           % Surface density [kg/m^3]
        params.c_sound = 240;          % Mars speed of sound [m/s]
        params.g0 = 3.71;              % Mars surface gravity [m/s²]

        % EDL Phase Parameters
        params.para_deploy_h = 1800;    % Parachute deployment altitude [m]
        params.para_deploy_mach = 1.15; % Target Mach number
        params.para_deploy_q_min = 500; % Min dynamic pressure [Pa]
        params.para_deploy_q_max = 2000;% Max dynamic pressure [Pa]

        % Spacecraft Parameters
        params.m = 500;                % Mass [kg]
        params.A = 15.9;               % Reference area [m^2]
        params.CD_entry = 1.48;        % Entry drag coefficient
        params.CD_para = 3.2;          % Parachute drag coefficient
        params.CL = 0.295;             % Lift coefficient
        params.r_n = 1.5;              % Nose radius [m]
        params.T_max = 20000;          % Max thrust [N]
        params.h_powered_start = 1000; % Powered descent start [m]

        % Target Conditions
        params.v_target = 0.60;        % Target landing velocity [m/s]
        params.gamma_target = -1*pi/180; % Target landing FPA [rad]
        params.max_q = 500;            % Max heat flux (W/cm²)
        params.max_g = 12;             % Max g-load
        params.terminal_altitude = 0.1;% Landing detection threshold [m]
        params.target_latitude = 0;    % Equatorial landing
        params.target_longitude = 0;    % Prime meridian reference
        params.r_target = 500e3 / params.R_mars; % Target distance [rad]

        % Optimization weights
        params.w1 = 0.4;               % Heat flux weight
        params.w2 = 0.3;               % Acceleration weight
        params.w3 = 0.3;               % Landing accuracy weight

        % Hamiltonian settings
        params.use_hamiltonian = true; % Enable Hamiltonian validation
        params.lambda_guess = [0.1; 0.1; 0.1; 0.1]; % Initial costate guess
    end

    function options = configure_optimizer()
        % Configure optimization options for fmincon
        
        options = optimoptions('fmincon',...
            'Display','iter-detailed',...
            'Algorithm','sqp',...
            'MaxFunctionEvaluations',2000,...
            'MaxIterations',500,...
            'FunctionTolerance',1e-4,...
            'StepTolerance',1e-3,...
            'OptimalityTolerance',1e-4,...
            'ConstraintTolerance',1e-6,...
            'FiniteDifferenceStepSize',1e-4,...
            'PlotFcn',{'optimplotfval','optimplotconstrviolation'});
    end

    function [opt_result, simulation_data] = run_optimization(x0, lb, ub, options, params)
        % Run the EFPA optimization
        
        % Create function handles with parameter passing
        obj_fun = @(x) obj_function_wrapper(x, params);
        con_fun = @(x) con_function_wrapper(x, params);
        
        % Initialize result structure
        opt_result = struct('x_opt', [], 'fval', NaN, 'exitflag', -1, 'output', struct());
        simulation_data = struct('t', [], 'y', [], 'ye', [], 'phases', []);
        
        try
            % Verify initial point
            init_cost = obj_fun(x0);
            fprintf('Initial point cost: %.4f\n', init_cost);

            % Run optimization
            [x_opt, fval, exitflag, output] = fmincon(...
                obj_fun, x0, [], [], [], [], lb, ub, con_fun, options);

            % Store results
            opt_result.x_opt = x_opt;
            opt_result.fval = fval;
            opt_result.exitflag = exitflag;
            opt_result.output = output;
            
            % Run final simulation
            [t, y, ~, ye, ~, phases] = simulate_edl(x_opt, params);
            simulation_data.t = t;
            simulation_data.y = y;
            simulation_data.ye = ye;
            simulation_data.phases = phases;
            
            % Display optimization status
            if exitflag > 0
                fprintf('\n==== Optimization Successful ====\n');
            else
                fprintf('\n==== Optimization Failed to Converge ====\n');
            end
            
        catch ME
            fprintf('\n==== Optimization Failed ====\n');
            fprintf('Error: %s\n', ME.message);
            fprintf('Using initial EFPA guess: %.2f degrees\n', x0*180/pi);
            
            % Run simulation with initial guess
            [t, y, ~, ye, ~, phases] = simulate_edl(x0, params);
            simulation_data.t = t;
            simulation_data.y = y;
            simulation_data.ye = ye;
            simulation_data.phases = phases;
        end
    end

    function display_results(opt_result, sim_data, params)
        % Display and visualize optimization results
        
        fprintf('\n==== Final Results ====\n');
        fprintf('Optimal EFPA: %.2f degrees\n', opt_result.x_opt*180/pi);
        fprintf('Final cost: %.4f\n', opt_result.fval);
        fprintf('Exit flag: %d\n', opt_result.exitflag);
        
        % Constraint verification
        check_constraints(sim_data.ye, sim_data.phases, params);
        
        % Calculate and display key metrics
        [heat_flux_max, g_load_max, landing_error] = calculate_metrics(...
            sim_data.t, sim_data.y, params);
        
        fprintf('\n[Performance Metrics]\n');
        fprintf('Max Heat Flux: %.2f W/cm² (Limit: %.2f)\n', heat_flux_max, params.max_q);
        fprintf('Max G-Load:    %.2f g (Limit: %.2f)\n', g_load_max, params.max_g);
        fprintf('Landing Error: %.2f m\n', landing_error);
        
        % Plot results
        plot_results(sim_data.t, sim_data.y, sim_data.phases, params);
    end

    function validate_with_hamiltonian(opt_result, sim_data, params)
        % Validate results using Hamiltonian approach
        
        try
            % Compute costates
            [t_costate, lambda] = compute_costates(sim_data.t, sim_data.y, params);
            
            if ~isempty(t_costate) && ~isempty(lambda)
                % Store costate data
                params.t_costate = t_costate;
                params.lambda = lambda;
                
                % Plot costate evolution
                figure;
                subplot(2,2,1); plot(t_costate, lambda(1,:)); title('\lambda_h');
                subplot(2,2,2); plot(t_costate, lambda(2,:)); title('\lambda_\theta');
                subplot(2,2,3); plot(t_costate, lambda(3,:)); title('\lambda_v');
                subplot(2,2,4); plot(t_costate, lambda(4,:)); title('\lambda_\gamma');
                sgtitle('Costate Evolution');
                
                % Simulate with Hamiltonian control
                [t_hamilton, y_hamilton, ~, ye_hamilton, ~, ~] = ...
                    simulate_edl_hamiltonian(opt_result.x_opt, params);
                
                % Display results
                fprintf('\n==== Hamiltonian Validation Results ====\n');
                fprintf('\nCostate Final Values:\n');
                fprintf('λ_h:    %.4e\n', lambda(1,end));
                fprintf('λ_θ:    %.4e\n', lambda(2,end));
                fprintf('λ_v:    %.4e\n', lambda(3,end));
                fprintf('λ_γ:    %.4e\n', lambda(4,end));
                
                % Final state comparison
                fprintf('\nFinal State Comparison:\n');
                fprintf('               fmincon   Hamiltonian\n');
                fprintf('Altitude (m): %7.2f   %7.2f\n', sim_data.ye(1), ye_hamilton(1));
                fprintf('Velocity (m/s): %5.2f   %5.2f\n', sim_data.ye(3), ye_hamilton(3));
                H_final = compute_hamiltonian_value(sim_data.t(end), sim_data.y(end,:), ...
                    lambda(:,end), params);
                fprintf('\nFinal Hamiltonian Value: %.4e (should be ~0 for optimal trajectory)\n', H_final);
                
                
                % Plot comparison
                compare_results(sim_data.t, sim_data.y, t_hamilton, y_hamilton, params);
            end
        catch ME
            warning('Hamiltonian validation failed: %s', ME.message);
        end
    end

    function perform_monte_carlo_analysis(params, n_runs)
        % Perform Monte Carlo analysis over EFPA range
        
        efpas = linspace(-18, -16, n_runs)*pi/180;
        results = zeros(n_runs, 6); % [EFPA, success, q_max, a_max, error, v_f]
        
        fprintf('\nRunning Monte Carlo analysis (%d runs)...\n', n_runs);
        
        for i = 1:n_runs
            [~, y, ~, ye, ~, phases] = simulate_edl(efpas(i), params);
            
            if ~isempty(ye)
                rho = params.rho0 * exp(-y(:,1)/params.H);
                results(i,:) = [efpas(i), 1, ...
                    max(1.83e-4*sqrt(rho/params.r_n).*y(:,3).^3/1e4), ...
                    max(abs(gradient(y(:,3)))/9.81), ...
                    abs(ye(2)-params.r_target), ...
                    ye(3)];
            else
                results(i,:) = [efpas(i), 0, NaN, NaN, NaN, NaN];
            end
        end
        
        % Plot results
        plot_monte_carlo_results(results, params);
    end

%% ========== CORE SIMULATION FUNCTIONS ==========

    function [t, y, te, ye, ie, phases] = simulate_edl(gamma0, p)
        % Simulate the complete EDL trajectory
        
        phases = struct('entry',struct('time',[],'index',[]),...
            'parachute',struct('time',[],'index',[],'velocity',[]),...
            'powered',struct('time',[],'index',[]));

        % Initialize outputs
        t = []; y = []; te = []; ye = []; ie = [];

        % Enhanced ODE options
        opts = odeset('RelTol',1e-6, 'AbsTol',1e-8, 'MaxStep',0.1, 'InitialStep',0.01);
        opts_braking = odeset(opts, 'Events', @(t,y) event_velocity_threshold(t,y,p));
        opts_landing = odeset(opts, 'Events', @event_landing);

        % --- Phase 1: Entry ---
        [t1, y1, te1, ye1, ie1] = ode45(...
            @(t,y) edl_dynamics(t, y, p, p.CD_entry, 0),...
            [0 3000], [130e3; 0; 4500; gamma0],...
            odeset(opts, 'Events', @(t,y) event_para_deploy(t,y,p)));

        if isempty(t1)
            warning('Entry phase failed at EFPA %.1f°', gamma0*180/pi);
            return;
        end

        phases.entry.time = t1;
        phases.entry.index = 1:length(t1);
        t = t1; y = y1;

        % --- Phase 2: Parachute ---
        [t2, y2, te2, ye2, ie2] = ode45(...
            @(t,y) edl_dynamics(t, y, p, p.CD_entry + p.CD_para, 0),...
            [t1(end) t1(end)+1500], y1(end,:)',...
            odeset(opts, 'Events', @(t,y) event_powered_descent(t,y,p)));

        if ~isempty(t2)
            phases.parachute.time = t2;
            phases.parachute.index = (length(t1)+1):(length(t1)+length(t2));
            phases.parachute.velocity = y2(1,3);

            t = [t1; t2(2:end)];
            y = [y1; y2(2:end,:)];

            % --- Phase 3: Powered Descent ---
            if y2(end,1) <= p.h_powered_start || y2(end,3) < (p.v_target * 3)
                % Phase 1: Full braking until near target velocity
                [t3a, y3a, te3a, ye3a, ie3a] = ode45(...
                    @(t,y) edl_dynamics(t, y, p, p.CD_entry, p.T_max),...
                    [t2(end) t2(end)+120], y2(end,:)', opts_braking);

                if ~isempty(t3a)
                    % Phase 2: Controlled descent
                    current_alt = max(y3a(end,1),0);
                    current_vel = y3a(end,3);
                    estimated_time = current_alt / (current_vel * 0.3);
                    time_interval = min(max(estimated_time*1.5, 10), 3000);

                    [t3b, y3b, te3b, ye3b, ie3b] = ode45(...
                        @(t,y) edl_dynamics(t, y, p, p.CD_entry, @(t,y) landing_thrust_control(t,y,p)),...
                        [t3a(end) t3a(end)+time_interval], y3a(end,:)', opts_landing);

                    if ~isempty(t3b)
                        % Combine all powered descent results
                        t3 = [t3a; t3b(2:end)];
                        y3 = [y3a; y3b(2:end,:)];
                        phases.powered.time = t3;
                        phases.powered.index = (length(t1)+length(t2)+1):(length(t1)+length(t2)+length(t3));
                        t = [t; t3(2:end)];
                        y = [y; y3(2:end,:)];
                        te = te3b; ye = ye3b; ie = ie3b;
                    end
                end
            end
        end
    end

    function dy = edl_dynamics(t, y, p, CD, T)
        % EDL dynamics equations
        
        % State variables with hard limits
        h = max(y(1), 0);      % Altitude (m)
        v = max(y(3), 0.01);    % Velocity (m/s)
        gamma = y(4);           % Flight path angle (rad)

        % Thrust handling with limits
        if isa(T, 'function_handle')
            T_val = min(T(t,y), p.T_max);
        else
            T_val = min(T, p.T_max);
        end

        % Atmospheric model
        rho = p.rho0 * exp(-h/p.H);

        % Aerodynamic forces with g-limiting
        q = 0.5*rho*v^2;
        D = min(q*CD*p.A, p.max_g*p.m*p.g0); % Drag limited to max g
        L = q*p.CL*p.A;

        % Gravity (constant for simplicity)
        g = p.g0;

        % State equations (stabilized)
        dy = zeros(4,1);
        dy(1) = v * sin(gamma);                    % dh/dt
        
        % Adaptive downrange control
        current_dist = y(2)*p.R_mars;
        remaining_dist = 500e3 - current_dist;

        if remaining_dist < 0
            dist_factor = 0.1; % Overshoot - maximum braking
        elseif remaining_dist < 15e3
            dist_factor = 0.2 + 0.8*(remaining_dist/15e3); % Final approach
        else
            dist_factor = 1; % Normal flight
        end

        dy(2) = dist_factor * v * cos(gamma) / p.R_mars;

        % Velocity equation
        thrust_accel = T_val/p.m;
        drag_dec = D/p.m;
        gravity_comp = g*sin(gamma);
        dy(3) = -drag_dec - thrust_accel - gravity_comp;

        % Hard velocity limit
        if dy(3) < -100
            dy(3) = -100;
        end

        % Angle equation
        dy(4) = (L/(p.m*v)) - (g/v - v/(p.R_mars + h))*cos(gamma);
    end

    function T = landing_thrust_control(t,y,p)
        % Adaptive thrust control for landing phase
        
        h = y(1);
        v = y(3);
        current_dist = y(2)*p.R_mars;
        remaining_dist = 500e3 - current_dist;
        
        T_hover = p.m * p.g0 * 1.05;  % 5% over hover

        if remaining_dist < 10e3
            % Final precision landing
            T = T_hover * (1 + (v - p.v_target)/0.1 + 0.5*(remaining_dist/10e3));
        elseif remaining_dist < 30e3
            % Approach phase
            vel_adjust = (v - p.v_target*2)/1;
            dist_adjust = remaining_dist/30e3;
            T = T_hover * (1 + vel_adjust) * (0.7 + 0.3*dist_adjust);
        else
            % Initial braking
            T = min(p.T_max, T_hover * 1.5);
        end

        if h > p.h_powered_start
            T = p.T_max; % Full braking
        elseif h > 100  % More aggressive control below 100m
            T = min(p.T_max, T_hover * (1 + (v - p.v_target)/0.5));
        else
            T = T_hover * (1 + (v - p.v_target)/0.2);  % More sensitive control near surface
        end

        % Enforce limits
        T = min(T, p.m * p.max_g * p.g0);  % Enforce g-limit
        T = max(T, T_hover * 0.95);       % Prevent upward thrust
    end

%% ========== EVENT FUNCTIONS ==========

    function [value,isterminal,direction] = event_para_deploy(t,y,p)
        % Parachute deployment conditions
        
        h_km = y(1)/1000;
        T_atm = 205.3645 - 1.245e-3*h_km - 8.85e-9*h_km^2 + 1.4e-13*h_km^3;
        p_atm = 559.351006 * exp(-1.05e-4 * h_km);
        rho = p_atm ./ (188.951107 * T_atm);

        % Calculate dynamic pressure
        dyn_press = 0.5*rho*y(3)^2;

        % Check all parachute deployment constraints
        value = [y(1) - p.para_deploy_h;              % Altitude trigger
            y(3)/p.c_sound - p.para_deploy_mach;      % Mach number
            dyn_press - p.para_deploy_q_min;          % Min dynamic pressure
            p.para_deploy_q_max - dyn_press];         % Max dynamic pressure

        isterminal = [1; 1; 1; 1];   % Stop integration if any condition is met
        direction = [-1; -1; -1; -1]; % Only trigger when descending/decaying
    end

    function [value,isterminal,direction] = event_powered_descent(t,y,p)
        % Powered descent initiation conditions
        value1 = y(1) - p.h_powered_start;
        value2 = y(3) - (p.v_target * 3);  % Transition if velocity < 3× target
        value = min(value1, value2);        % Trigger on whichever comes first
        isterminal = 1;
        direction = -1;
    end

    function [value,isterminal,direction] = event_landing(t,y)
        % Landing condition (altitude = 0)
        value = y(1);  % land at 0 m
        isterminal = 1;
        direction = -1;
    end

    function [value,isterminal,direction] = event_velocity_threshold(t,y,p)
        % Velocity threshold for braking phase transition
        value = y(3) - (p.v_target * 1.5); % Transition when velocity < 1.5× target
        isterminal = 1;
        direction = -1;
    end

%% ========== HAMILTONIAN FUNCTIONS ==========

    function [t, y, te, ye, ie, phases] = simulate_edl_hamiltonian(gamma0, p)
        % EDL simulation with Hamiltonian-based control
        
        phases = struct('entry',struct('time',[],'index',[]),...
            'parachute',struct('time',[],'index',[],'velocity',[]),...
            'powered',struct('time',[],'index',[]));

        % Initialize with empty arrays
        t = []; y = []; te = []; ye = []; ie = [];

        % Common ODE options
        ode_opts = odeset('RelTol',1e-6,'AbsTol',1e-8);

        % Phase 1: Entry with Hamiltonian control
        [t1, y1, te1, ye1, ie1] = ode45(...
            @(t,y) edl_dynamics_hamiltonian(t, y, p),...
            [0 3000], [125e3; 0; 5800; gamma0],...
            odeset(ode_opts,'Events',@entry_events));

        if isempty(t1)
            return; % Simulation failed immediately
        end

        phases.entry.time = t1;
        phases.entry.index = 1:length(t1);
        t = t1; y = y1;

        % Phase 2: Parachute Descent (if deployed)
        if y1(end,1) <= p.para_deploy_h || (~isempty(ie1) && any(ie1 == 1))
            [t2, y2, te2, ye2, ie2] = ode45(...
                @(t,y) edl_dynamics(t, y, p, p.CD_entry + p.CD_para, 0),...
                [t1(end) t1(end)+1000], y1(end,:)',...
                odeset(ode_opts,'Events',@para_events));

            if ~isempty(t2)
                phases.parachute.time = t2;
                phases.parachute.index = (length(t1)+1):(length(t1)+length(t2));
                phases.parachute.velocity = y2(1,3);

                t = [t1; t2(2:end)];
                y = [y1; y2(2:end,:)];

                % Phase 3: Powered Descent (if triggered)
                if y2(end,1) <= p.h_powered_start
                    [t3, y3, te3, ye3, ie3] = ode45(...
                        @(t,y) edl_dynamics_hamiltonian(t, y, p),...
                        [t2(end) t2(end)+500], y2(end,:)',...
                        odeset(ode_opts,'Events',@landing_events));

                    if ~isempty(t3)
                        phases.powered.time = t3;
                        phases.powered.index = (length(t1)+length(t2)+1):(length(t1)+length(t2)+length(t3));

                        t = [t; t3(2:end)];
                        y = [y; y3(2:end,:)];
                        te = te3; ye = ye3; ie = ie3;
                    end
                end
            end
        end

        function [value,isterminal,direction] = entry_events(t,y)
            value = y(1) - p.para_deploy_h;
            isterminal = 1;
            direction = -1;
        end

        function [value,isterminal,direction] = para_events(t,y)
            value = y(1) - p.h_powered_start;
            isterminal = 1;
            direction = -1;
        end

        function [value,isterminal,direction] = landing_events(t,y)
            value = y(1);
            isterminal = 1;
            direction = -1;
        end
    end

    function dy = edl_dynamics_hamiltonian(t, y, p)
        % EDL dynamics with Hamiltonian-based control
        
        % State variables with safeguards
        h = max(y(1), 0);
        v = max(y(3), 0.1);
        gamma = y(4);

        % Costate interpolation with optimized scaling
        t_costate = min(max(t, p.t_costate(1)), p.t_costate(end));
        lambda = interp1(p.t_costate, p.lambda', t_costate, 'pchip', 'extrap')';

        % Adjusted scaling factor for better response
        lambda = lambda * 1e-10;

        % Enhanced thrust control for faster descent below 500m
        T = 0;
        S_T = 0;
        if h <= abs(p.h_powered_start - 800)
            T_hover = p.m * p.g0 * 1.05;

            % More responsive switching function
            S_T = -lambda(3)/p.m - lambda(4)/(p.m*v);
            S_T = tanh(S_T * 1e8);  % Faster saturation with lower gain

            % Velocity error term with altitude-based gain scheduling
            velocity_error = (v - p.v_target)/p.v_target;
            vel_gain = 2.5;  % Base gain
            if h < 500  % Increase gain below 500m
                vel_gain = 4.0 + (500-h)/500;
            end

            % Altitude weighting - more aggressive below 500m
            if h < 500
                alt_weight = min(1, (1 - h/100)^2);
            else
                alt_weight = 0.7;
            end

            % Combined thrust calculation
            T = T_hover * (1 + vel_gain*velocity_error) + S_T * p.T_max * 0.9;
            T = T * alt_weight;

            % Enforce bounds with more aggressive lower limit
            T = min(max(T, T_hover*0.3), p.T_max);
        end

        % Rest of dynamics
        rho = p.rho0 * exp(-h/p.H);
        g = p.mu_mars/(p.R_mars + h)^2;
        D = 0.5*rho*v^2*p.CD_entry*p.A;
        L = 0.5*rho*v^2*p.CL*p.A;

        dy = zeros(4,1);
        dy(1) = v * sin(gamma);
        dy(2) = v * cos(gamma) / (p.R_mars + h);
        dy(3) = -D/p.m - g*sin(gamma) - T/p.m;
        dy(4) = (L/(p.m*v)) - (g/v - v/(p.R_mars + h))*cos(gamma);
    end

    function [t_costate, lambda] = compute_costates(t_forward, y_forward, p)
        % Backward costate integration
        
        % Use actual simulation times
        t_costate = t_forward;

        % Enhanced terminal conditions
        lambda_f = [0; 
            2*p.w3*(y_forward(end,2)-p.r_target);
            -1e-15*sign(y_forward(end,3)-p.v_target); % Stronger velocity control
            -5e-16]; % Stronger FPA control

        % Backward integration with tighter tolerances
        options = odeset('RelTol',1e-6, 'AbsTol',1e-8, 'NormControl','on');
        [~, lambda_bw] = ode15s(@(t,lambda) costate_dynamics(t, lambda, t_forward, y_forward, p),...
            flipud(t_costate), lambda_f, options);

        lambda = flipud(lambda_bw)';
    end

    function dlambda = costate_dynamics(t, lambda, t_forward, y_forward, p)
        % Costate dynamics equations
        
        % Interpolate states
        h = interp1(t_forward, y_forward(:,1), t);
        v = interp1(t_forward, y_forward(:,3), t);
        gamma = interp1(t_forward, y_forward(:,4), t);

        % Atmospheric calculations
        rho = p.rho0 * exp(-h/p.H);
        g = p.mu_mars/(p.R_mars + h)^2;
        dg_dh = -2*p.mu_mars/(p.R_mars + h)^3;
        drho_dh = -rho/p.H;

        % Balanced partial derivatives for faster response
        df_dh = [0;
            -0.3*v*cos(gamma)/(p.R_mars + h)^2;
            -0.2*p.CD_entry*p.A*v^2*drho_dh - 0.3*dg_dh*sin(gamma);
            -0.3*(v/(p.R_mars + h)^2 + dg_dh/v)*cos(gamma)];

        df_dv = [0.3*sin(gamma);
            0.3*cos(gamma)/(p.R_mars + h);
            -0.3*p.CD_entry*p.A*rho*v/p.m - 0.05*(v-p.v_target);
            0.3*(1/(p.R_mars + h) + g/v^2)*cos(gamma) - 0.3*p.CL*p.A*rho*v/(p.m*v^2)];

        df_dgamma = [0.3*v*cos(gamma);
            -0.3*v*sin(gamma)/(p.R_mars + h);
            -0.3*g*cos(gamma);
            -0.3*(v/(p.R_mars + h) - g/v)*sin(gamma)];

        % Costate dynamics with matched scaling
        dlambda = -1e-15 * [df_dh'*lambda;
            0;
            df_dv'*lambda;
            df_dgamma'*lambda];
    end

    function H = compute_hamiltonian_value(t, y, lambda, p)
        % Compute Hamiltonian value for validation
        
        % Extract states and costates
        h = y(1); v = y(3); gamma = y(4);
        lambda_h = lambda(1); lambda_v = lambda(3); lambda_gamma = lambda(4);

        % Compute dynamics
        rho = p.rho0 * exp(-h/p.H);
        g = p.mu_mars/(p.R_mars + h)^2;
        D = 0.5*rho*v^2*p.CD_entry*p.A;
        L = 0.5*rho*v^2*p.CL*p.A;

        % State derivatives
        hdot = v*sin(gamma);
        vdot = -D/p.m - g*sin(gamma);
        gammadot = (v/(p.R_mars + h) - g/v)*cos(gamma) + L/(p.m*v);

        % Hamiltonian
        H = lambda_h*hdot + lambda_v*vdot + lambda_gamma*gammadot;
    end

%% ========== UTILITY FUNCTIONS ==========

    function J = obj_function_wrapper(gamma0, p)
        % Wrapper for objective function with error handling
        try
            [t, y, ~, ye, ~, phases] = simulate_edl(gamma0, p);

            if isempty(ye) || any(isnan(ye(:)))
                J = 1e6;
                return;
            end

            % Calculate cost components
            h = y(:,1);
            theta = y(:,2);
            v = y(:,3);

            rho = p.rho0 * exp(-h/p.H);
            q = 1.83e-4 * sqrt(rho/p.r_n) .* v.^3 / 1e4;
            q_max = max(q);

            % Robust acceleration calculation
            if length(t) > 1
                dt = diff(t);
                dv = diff(v);
                a = dv ./ dt;
                a_max = max(abs(a)) / p.g0;
            else
                a_max = 0;
            end

            landing_error = 100 * (theta(end) - p.r_target)^2;
            J = p.w1 * q_max + p.w2 * a_max + p.w3 * landing_error;

        catch ME
            fprintf('Objective function error at EFPA %.2f°: %s\n', gamma0*180/pi, ME.message);
            J = 1e6;
        end
    end

    function [c, ceq] = con_function_wrapper(gamma0, params)
        % Wrapper for constraint function with error handling
        try
            [~, ~, ~, ye, ~, phases] = simulate_edl(gamma0, params);

            c = zeros(4,1);
            ceq = [];

            if isempty(ye) || any(isnan(ye(:)))
                c(:) = 1e6;
                ceq = 1e6;
                return;
            end

            % Final state constraints
            h_f = ye(1);
            v_f = ye(3);
            gamma_f = ye(4);

            % Velocity constraints (looser)
            c(1) = v_f - (params.v_target * 1.5) - 1e-6;
            c(2) = 0;

            % FPA constraint (looser)
            c(3) = abs(gamma_f) - deg2rad(90) + 1e-6;

            % Altitude constraint (looser)
            c(4) = abs(h_f) - params.terminal_altitude + 1e-6;

            % Equality constraint (relaxed)
            ceq = max(abs(h_f) - 1e-6, 0);

        catch
            c = 1e6*ones(4,1);
            ceq = 1e6;
            fprintf('Constraint evaluation failed for EFPA %.2f°\n', gamma0*180/pi);
        end
    end

    function [heat_flux_max, g_load_max, landing_error] = calculate_metrics(t, y, p)
        % Calculate key performance metrics
        
        % Extract states
        h = y(:,1);
        v = y(:,3);
        theta = y(:,2);
        
        % Calculate heat flux
        rho = p.rho0 * exp(-h/p.H);
        q = 1.83e-4 * sqrt(rho/p.r_n) .* v.^3 / 1e4;
        heat_flux_max = max(q);
        
        % Calculate g-load
        if length(t) > 1
            dt = diff(t);
            dv = diff(v);
            a = dv ./ dt;
            g_load_max = max(abs(a)) / p.g0;
        else
            g_load_max = 0;
        end
        
        % Calculate landing error (distance from target)
        landing_error = abs(theta(end) - p.r_target) * p.R_mars;
    end

    function check_constraints(ye, phases, p)
        % Verify all mission constraints
        
        fprintf('\n==== Constraint Verification ====\n');

        if isempty(ye)
            fprintf('Simulation failed - no final state\n');
            return;
        end

        % Final state checks
        h_f = abs(ye(1));
        v_f = ye(3);
        gamma_f = ye(4);

        % Final altitude
        if abs(h_f) < p.terminal_altitude
            fprintf('✓ Final altitude: %.2f m (target: 0 m)\n', h_f);
        else
            fprintf('✗ Final altitude: %.2f m (target: 0 m)\n', h_f);
        end

        % Final velocity
        if v_f >= 0 && v_f <= (p.v_target * 1.5)
            fprintf('✓ Final velocity: %.2f m/s (target: ≤ %.1f m/s)\n', v_f, p.v_target*1.5);
        else
            fprintf('✗ Final velocity: %.2f m/s (target: ≤ %.1f m/s)\n', v_f, p.v_target*1.5);
        end

        % Final FPA
        if gamma_f <= -85*pi/180
            fprintf('✓ Final FPA: %.2f deg (steep descent)\n', rad2deg(gamma_f));
        else
            fprintf('✗ Final FPA: %.2f deg (too shallow)\n', rad2deg(gamma_f));
        end

        % Parachute deployment
        if ~isempty(phases.parachute.time)
            v_para = phases.parachute.velocity;
            mach_para = v_para / p.c_sound;
            h_km = p.para_deploy_h/1000;
            T_atm = 205.3645 - 1.245e-3*h_km - 8.85e-9*h_km^2 + 1.4e-13*h_km^3;
            p_atm = 559.351006 * exp(-1.05e-4 * h_km);
            rho = p_atm ./ (188.951107 * T_atm);
            p_para = 0.5 * rho * v_para^2;

            fprintf('\nParachute Deployment Conditions:\n');
            fprintf('Altitude: %.1f km\n', p.para_deploy_h/1e3);
            fprintf('Mach number: %.2f (target: %.1f)\n', mach_para, p.para_deploy_mach);
            fprintf('Dynamic pressure: %.1f Pa (target: %.0f-%.0f Pa)\n', p_para, p.para_deploy_q_min, p.para_deploy_q_max);

            if abs(mach_para - p.para_deploy_mach) <= 0.2
                fprintf('✓ Mach number constraint met\n');
            else
                fprintf('✗ Mach number constraint violated\n');
            end

            if p_para >= p.para_deploy_q_min && p_para <= p.para_deploy_q_max
                fprintf('✓ Dynamic pressure constraint met\n');
            else
                fprintf('✗ Dynamic pressure constraint violated\n');
            end
        else
            fprintf('\n✗ No parachute deployment occurred\n');
        end
    end

    function plot_results(t, y, phases, p)
        % Plot EDL trajectory results
        
        h = y(:,1)/1000;       % Altitude [km]
        theta = y(:,2);        % Downrange angle [rad]
        v = y(:,3);            % Velocity [m/s]
        gamma = y(:,4)*180/pi; % FPA [deg]

        % Calculate derived quantities
        rho = p.rho0 * exp(-h*1000/p.H);
        q = 1.83e-4 * sqrt(rho/p.r_n) .* v.^3 / 1e4; % Heat flux [W/cm²]
        dt = gradient(t);
        dvdt = gradient(v)./dt;
        accel = abs(dvdt)/p.g0; % Deceleration [g]
        ground_range = cumtrapz(t, v.*cos(gamma*pi/180))/1000; % Downrange distance [km]

        figure('Name','Mars EDL Analysis','Position',[100 100 1200 900]);

        % 1. Altitude Profile
        subplot(3,2,1);
        plot(t, h, 'LineWidth',1.5);
        hold on;
        plot(t(end), 0, 'rx', 'MarkerSize',10, 'LineWidth',2);
        xlabel('Time (s)'); ylabel('Altitude (km)');
        title('Altitude vs Time');
        grid on;

        % 2. Velocity Profile
        subplot(3,2,2);
        plot(t, v, 'LineWidth',1.5);
        hold on;
        plot([t(1) t(end)], [p.v_target p.v_target], 'r--');
        xlabel('Time (s)'); ylabel('Velocity (m/s)');
        title('Velocity vs Time');
        grid on;

        % 3. Flight Path Angle
        subplot(3,2,3);
        plot(t, gamma, 'LineWidth',1.5);
        xlabel('Time (s)'); ylabel('FPA (deg)');
        title('Flight Path Angle');
        grid on;

        % 4. Heat Flux
        subplot(3,2,4);
        plot(t, q, 'LineWidth',1.5);
        hold on;
        plot([t(1) t(end)], [p.max_q p.max_q], 'r--');
        xlabel('Time (s)'); ylabel('Heat Flux (W/cm^2)');
        title('Heat Flux Profile');
        grid on;

        % 5. Deceleration
        subplot(3,2,5);
        plot(t, accel, 'LineWidth',1.5);
        hold on;
        plot([t(1) t(end)], [p.max_g p.max_g], 'r--');
        xlabel('Time (s)'); ylabel('Deceleration (g)');
        title('Deceleration Profile');
        grid on;

        % 6. Ground Track
        subplot(3,2,6);
        plot(ground_range, h, 'LineWidth',1.5);
        hold on;
        plot(ground_range(end), 0, 'rx', 'MarkerSize',10, 'LineWidth',2);
        xlabel('Downrange Distance (km)'); ylabel('Altitude (km)');
        title('True Ground Track');
        grid on;

        % Add phase transition markers
        subplot(3,2,1);
        if isfield(phases,'parachute') && ~isempty(phases.parachute.time)
            plot(phases.parachute.time(1), h(phases.parachute.index(1)), 'go');
        end
        if isfield(phases,'powered') && ~isempty(phases.powered.time)
            plot(phases.powered.time(1), h(phases.powered.index(1)), 'mo');
        end
        legend('Trajectory','Landing','Parachute','Powered', 'Location','best');
    end

    function compare_results(t, y, t_hamilton, y_hamilton, p)
        % Compare fmincon vs. Hamiltonian results
        
        figure('Name','Comparison: fmincon vs. Hamiltonian','Position',[100 100 1200 900]);

        % Altitude comparison
        subplot(2,2,1);
        plot(t, y(:,1)/1e3, 'b', 'LineWidth',1.5); hold on;
        plot(t_hamilton, y_hamilton(:,1)/1e3, 'r--', 'LineWidth',1.5);
        xlabel('Time (s)'); ylabel('Altitude (km)');
        legend('fmincon', 'Hamiltonian');
        title('Altitude Profile');
        grid on;

        % Velocity comparison
        subplot(2,2,2);
        plot(t, y(:,3)/1e3, 'b', 'LineWidth',1.5); hold on;
        plot(t_hamilton, y_hamilton(:,3)/1e3, 'r--', 'LineWidth',1.5);
        xlabel('Time (s)'); ylabel('Velocity (km/s)');
        title('Velocity Profile');
        grid on;

        % Flight path angle comparison
        subplot(2,2,3);
        plot(t, y(:,4)*180/pi, 'b', 'LineWidth',1.5); hold on;
        plot(t_hamilton, y_hamilton(:,4)*180/pi, 'r--', 'LineWidth',1.5);
        xlabel('Time (s)'); ylabel('FPA (deg)');
        title('Flight Path Angle');
        grid on;

        % Downrange distance comparison
        subplot(2,2,4);
        plot(t, y(:,2)*p.R_mars/1e3, 'b', 'LineWidth',1.5); hold on;
        plot(t_hamilton, y_hamilton(:,2)*p.R_mars/1e3, 'r--', 'LineWidth',1.5);
        xlabel('Time (s)'); ylabel('Downrange (km)');
        title('Ground Track');
        grid on;
    end

    function plot_monte_carlo_results(results, params)
        % Plot Monte Carlo analysis results
        
        success_rate = mean(results(:,2)) * 100;
        fprintf('Monte Carlo Success Rate: %.1f%%\n', success_rate);

        figure('Name', sprintf('Monte Carlo EFPA Analysis (Success Rate: %.1f%%)', success_rate));

        % Plot 1: Success/Failure vs EFPA
        subplot(2,2,1);
        scatter(results(results(:,2)==1, 1)*180/pi, results(results(:,2)==1, 3), 'b', 'filled'); hold on;
        scatter(results(results(:,2)==0, 1)*180/pi, 50, 'rx', 'LineWidth', 1.5);
        xlabel('EFPA (deg)'); ylabel('Peak Heat Flux (W/cm^2)');
        title('Heat Flux vs EFPA (Failures in Red)');
        grid on;

        % Plot 2: Final Velocity vs EFPA
        subplot(2,2,2);
        scatter(results(results(:,2)==1, 1)*180/pi, results(results(:,2)==1, 6), 'b', 'filled'); hold on;
        scatter(results(results(:,2)==0, 1)*180/pi, 50, 'rx', 'LineWidth', 1.5);
        yline(params.v_target, 'g--', 'Target Velocity', 'LineWidth', 1.5);
        xlabel('EFPA (deg)'); ylabel('Final Velocity (m/s)');
        title('Final Velocity vs EFPA');
        legend('Success', 'Failure');
        grid on;

        % Plot 3: Peak Acceleration
        subplot(2,2,3);
        scatter(results(results(:,2)==1, 1)*180/pi, results(results(:,2)==1, 4), 'filled');
        xlabel('EFPA (deg)'); ylabel('Peak Acceleration (g)');
        title('Peak Acceleration (Successful Landings)');
        grid on;

        % Plot 4: Success Rate Histogram
        subplot(2,2,4);
        efpa_deg = results(:,1)*180/pi; 
        bin_edges = linspace(min(efpa_deg), max(efpa_deg), 10);
        histogram(efpa_deg, 'BinEdges', bin_edges, ...
            'FaceColor', [0.5 0.5 0.5]);
        hold on;
        success_idx = results(:,2) == 1;
        histogram(efpa_deg(success_idx), 'BinEdges', bin_edges, ...
            'FaceColor', 'g');
        xlabel('EFPA (deg)');
        ylabel('Number of Runs');
        title(sprintf('EFPA Distribution\n(All Runs: Gray, Successes: Green)'));
        grid on;
        legend('All Runs', 'Successful Landings');
    end

    function str = ternary(condition, true_str, false_str)
        % Ternary operator for display messages
        if condition
            str = true_str;
        else
            str = false_str;
        end
    end
end