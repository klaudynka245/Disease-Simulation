using Agents , Random , InteractiveDynamics , CairoMakie 
using DrWatson: @dict
#using DataFrames , Plots


mutable struct Agent2 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    days_infected::Int # how many days have passed after infection
    status::Symbol # :S (susceptible), :I (infecious), :R (removed, immune), :Q (quarantine and infecious) :V (vaccinated and removed)
    days_quarantine::Int
    hygiene::Float64
end

function if_susceptible(agent::Agent2)
    return agent.status == :S
end


function symulation(;
    infection_period = 24 ,
    detection_time = 14 ,
    reinfection_probability = 0.01,
    isolated = 0.2, # 0 is nobody is isolated , 1 eve ryone is isolated
    interaction_radius = 0.01,
    dt = 1,
    speed = 0.0012,
    death_rate = 0.044,
    N = 1000,
    initial_infected = 5,
    seed = 1410,
    hygiene_min = 0.4,
    hygiene_max = 0.8,
    time_of_quarantine = 10,
    steps_per_day = 12,
    chance_to_go_quaratine = 0.5,
    symulation_time = 0,
    vaccine_per_day = 2
    )

    infection_period *= steps_per_day
    detection_time *=  steps_per_day
    time_of_quarantine *= steps_per_day

    properties = @dict(
        infection_period,
        detection_time,
        reinfection_probability,
        death_rate,
        dt,
        interaction_radius,
        time_of_quarantine,
        speed,
        chance_to_go_quaratine,
        steps_per_day,
        symulation_time,
        N,
        vaccine_per_day)

    space = ContinuousSpace((1,1), 0.02)
    model = ABM(Agent2,space, properties = properties, rng = MersenneTwister(seed))

    # Add agents to the model
    for id in 1:N
        pos = Tuple(rand(model.rng, 2))
        status = id > initial_infected ? :S : :I
        is_isolated = isolated * N >= id
        mass = is_isolated ? Inf : 1.0
        vel = is_isolated ? (0,0) : sincos(2pi* rand(model.rng,)) .* speed
        hygiene = (hygiene_max-hygiene_min)*rand(model.rng) + hygiene_min
        add_agent!(pos,model,vel, mass, 0, status, 0, hygiene)
    end
    return model
end

function transmit!(a,b,reinfection,model)
    if  count(a.status == :I for a in (a,b)) in (0,2)
        return
    end
    
    infected, healthy = a.status == :I ? (a,b) : (b,a)
    rand(model.rng) < (infected.hygiene + healthy.hygiene)/2 && return 

   
    if healthy.status == :R || healthy.status == :V
        if rand(model.rng) > reinfection 
            return
        end
    end

    healthy.status = :I
end

function vaccine!(a,model)
    a.status = :V   
end

function model_step!(model)
    r = model.interaction_radius
    for (a, b) in interacting_pairs(model, r, :nearest)
        transmit!(a,b,model.reinfection_probability,model)
        elastic_collision!(a, b, :mass)
    end
    
    if model.symulation_time % model.steps_per_day == 0
        for _ in 1:model.vaccine_per_day
            agent = random_agent(model, if_susceptible)
            if agent != Nothing
                vaccine!(agent, model)
            else
                model.symulation_time += 1
                return
            end
        end
       
    end
    model.symulation_time += 1
end

function quarantine!(a,model)
    if a.status == :I && a.days_infected >= model.detection_time && rand(model.rng) < model.chance_to_go_quaratine / model.steps_per_day
        a.status = :Q
        a.vel = (0,0)
        a.mass = Inf

    end
end

function quarantine_end!(a,model)
    if a.days_quarantine == model.time_of_quarantine
        a.days_quarantine = 0
        recover_or_die!(a,model) 
        a.vel = sincos(2pi* rand(model.rng,)) .* model.speed
        a.mass = 1.0
    end
end

function recover_or_die!(agent, model)
    if agent.days_infected >= model.infection_period
        if rand(model.rng) <= model.death_rate
            kill_agent!(agent, model)
        else 
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

function agent_step!(agent,model)
    move_agent!(agent, model, model.dt)
    agent.status in (:I,:Q) ? agent.days_infected +=1 : agent.days_infected = 0
    agent.status == :Q ? agent.days_quarantine +=1 : agent.days_quarantine = 0
    recover_or_die!(agent, model)
    quarantine!(agent,model)
    quarantine_end!(agent,model)
end
sir_colors(a) = a.status == :S ? "#000000" : a.status == :I ? "#ff0000" : a.status == :Q ? "#00FFFF" : a.status == :V ? "#993399" : "#00FF00"

sir_model = symulation(isolated = 0.8 )
println("Done")

abm_video("vaccine_0.8.mp4",
    sir_model,
    agent_step!,
    model_step!,
    title = " Symulation_vac",
    ac = sir_colors,
    frames = 500 , spf = 2, framerate = 25)

println("Już___________________________________________")