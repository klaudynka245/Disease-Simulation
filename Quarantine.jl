using Agents , Random , InteractiveDynamics , CairoMakie 
using DrWatson: @dict


mutable struct Agent2 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    days_infected::Int # how many days have passed after infection
    status::Symbol # :S (susceptlibe), :I (infecious), :R (removed, immune), :Q (quarantine and infecious) 
    days_quarantine::Int
    hygiene::Float64
end


function symulation(;
    infection_period = 30 ,
    detection_time = 14 ,
    reinfection_probability = 0.05,
    isolated = 0.0, # 0 is nobady is isolated , 1 evryone is isolated
    interaction_radius = 0.012,
    dt = 1,
    speed = 0.002,
    death_rate = 0.044,
    N = 1000,
    initial_infected = 5,
    seed = 1410,
    hygiene_min = 0.4,
    hygiene_max = 0.8,
    time_of_quarantine = 10,
    steps_per_day = 24,
    chance_to_go_quaratine = 0.5
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
        steps_per_day)

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
    rand(model.rng) > infected.hygiene && return 

    #healthy.status == :R && rand(model.rng) > reinfection ? return : healthy.status = :I
    if healthy.status == :R
        if rand(model.rng) > reinfection 
            return
        end
    end

    healthy.status = :I
end

function model_step!(model)
    r = model.interaction_radius
    for (a, b) in interacting_pairs(model, r, :nearest)
        transmit!(a,b,model.reinfection_probability,model)
        elastic_collision!(a, b, :mass)
    end
end

function quarantine!(a,model)
    if a.status == :I && a.days_infected >= model.detection_time && rand(model.rng) > model.chance_to_go_quaratine / model.steps_per_day
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

sir_model = symulation(isolated = 0.8 )
sir_colors(a) = a.status == :S ? "#000000" : a.status == :I ? "#ff0000" : a.status == :Q ? "#00FFFF" : "#00FF00"

abm_video("Trzecia3.mp4",
sir_model,
agent_step!,
model_step!,
title = " Symulation",
ac = sir_colors,
frames = 500 , spf = 2, framerate = 25)
println("Ju??___________________________________________________________________")