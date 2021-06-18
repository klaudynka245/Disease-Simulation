using Agents , Random , InteractiveDynamics , CairoMakie 
using DrWatson: @dict
#using DataFrames , Plots


mutable struct Agent2 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    days_infected::Int # how many days have passed after infection
    status::Symbol # :S (susceptible), :I (infecious), :R (removed, immune), :Q (quarantine and infecious) 
    days_quarantine::Int
    hygiene::Float64
    carefull::Bool
end


function symulation(;
    infection_period = 24 ,
    detection_time = 14 ,
    reinfection_probability = 0.01,
    isolated = 0.2, # 0 is nobody is isolated , 1 everyone is isolated
    interaction_radius = 0.01,
    dt = 1,
    speed = 0.0012,
    death_rate = 0.044,
    N = 200,
    initial_infected = 5,
    seed = 1410,
    hygiene_min = 0.4,
    hygiene_max = 0.8,
    time_of_quarantine = 10,
    steps_per_day = 12,
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
        carefull = id > 0.1*N ? false : true
        add_agent!(pos,model,vel, mass, 0, status, 0, hygiene, carefull)
    end
    return model
end

function transmit!(a,b,reinfection,model)
    if  count(a.status == :I for a in (a,b)) in (0,2)
        return
    end
    
    infected, healthy = a.status == :I ? (a,b) : (b,a)
    rand(model.rng) < (infected.hygiene + healthy.hygiene)/2 && return 

   
    if healthy.status == :R
        if rand(model.rng) > reinfection 
            return
        end
    end

    healthy.status = :I
end

function avoid(a, model)
    if a.mass != Inf
        a.vel = sincos(2pi* rand(model.rng,)) .* model.speed
    end
end

function model_step!(model)
    r = model.interaction_radius
    r2 = 5*r
    for (a, b) in interacting_pairs(model, r, :nearest)
        transmit!(a,b,model.reinfection_probability,model)
        elastic_collision!(a, b, :mass)
    end
    for (a, b) in interacting_pairs(model, r2, :nearest)
        a.carefull == true  ? avoid(a, model) : continue  
    end

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