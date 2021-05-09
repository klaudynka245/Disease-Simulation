#https://www.contagionlive.com/view/analysis-spanish-flu-pandemic-proves-social-distancing-works
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2391305/
#https://www.britannica.com/event/influenza-pandemic-of-1918-1919
#https://www.cdc.gov/flu/symptoms/flu-vs-covid19.htm
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4954461/
#https://www.kochanski.pl/pl/koronawirus-na-tle-innych-pandemii-czy-okaze-sie-bardziej-smiercionosny-od-grypy-hiszpanki/
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2862334/
#https://www.saturdayeveningpost.com/2020/03/vintage-ads-selling-health-and-hygiene-during-the-1918-pandemic/

"""Grypa hiszpanka, 1918 r.
Podstawowa liczba odtwarzania: około 1,8
Wskaźnik śmiertelności: 2,5%
Liczba ludności na świecie: 1,8 mld (dane szacunkowe)
Grypa z 1918 roku była jedną z najstraszliwszych pandemii XX wieku, która według WHO szczególnie mocno dotknęła osoby 
w wieku 20-40 lat. COVID-19, o podstawowej liczbie odtwarzania wynoszącej 2, jest od niej nieco bardziej zaraźliwa.
Grypa z roku 1918, zwana „hiszpanką”, w rzeczywistości wcale nie pochodziła z Hiszpanii. Przy wskaźniku śmiertelności 
wynoszącym 2,5% choroba ta pochłonęła więcej ofiar śmiertelnych (30-50 mln) niż I Wojna Światowa z 20 mln ofiar. Jeżeli 
wynosząca 4,5% śmiertelności dla COVID-19 spadnie, skutki tej pandemii nie będą tak straszne, jak w przypadku hiszpanki. 
Jeśli jednak wskaźnik ten utrzyma się na obecnym poziomie, COVID-19 będzie prawie dwukrotnie groźniejsza."""

using Agents , Random , InteractiveDynamics , CairoMakie 
using DrWatson: @dict



mutable struct Agent2 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    days_infected::Int # how many days have passed after infection
    status::Symbol # :S (susceptlibe), :I (infecious), R (removed, immune)
    hygiene::Float64
end

const steps_per_day = 24

function symulation(;
    infection_period = 30 * steps_per_day,
    detection_time = 4 * steps_per_day,
    reinfection_probability = 0.05,
    isolated = 0.1, # 0 is nobady is isolated , 1 evryone is isolated
    interaction_radius = 0.012,
    dt = 1,
    speed = 0.002,
    death_rate = 0.025,
    N = 1000,
    initial_infected = 5,
    seed = 1410,
    hygiene_min = 0.1,
    hygiene_max = 0.5)

    properties = @dict(
        infection_period,
        detection_time,
        reinfection_probability,
        death_rate,
        dt,
        interaction_radius)

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
        add_agent!(pos,model,vel, mass, 0, status, hygiene)
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
    agent.status == :I ? agent.days_infected +=1 : agent.days_infected = 0
    recover_or_die!(agent, model)
end

sir_model = symulation(isolated = 0.8 )
sir_colors(a) = a.status == :S ? "#000000" : a.status == :I ? "#ff0000" : "#00FF00"

"""abm_video("Piąta.mp4",
sir_model,
agent_step!,
model_step!,
title = " Symulation",
ac = sir_colors,
frames = 500 , spf = 2, framerate = 25)
println("Już___________________________________________________________________")"""


infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)

adata = [(:status, infected), (:status, recovered)]

r1, r2 = 0.04, 0.33
sir_model1 = symulation(reinfection_probability = r1)
sir_model2 = symulation(reinfection_probability = r2)
sir_model3 = symulation(reinfection_probability = r1)

data1, _ = run!(sir_model1, agent_step!, model_step!, 500; adata)
data2, _ = run!(sir_model2, agent_step!, model_step!, 500; adata)
data3, _ = run!(sir_model3, agent_step!, model_step!, 500; adata)

data1[(end-10):end, :]

using CairoMakie
figure = Figure()
ax = figure[1, 1] = Axis(figure; ylabel = "Infected")
l1 = lines!(ax, data1[:, dataname((:status, infected))], color = :orange)
l2 = lines!(ax, data2[:, dataname((:status, infected))], color = :blue)
l3 = lines!(ax, data3[:, dataname((:status, infected))], color = :green)
figure[1, 2] =
    Legend(figure, [l1, l2, l3], ["r=$r1", "r=$r2", "r=$r1"])
figure

r4 = 0.04
sir_model4 = symulation(reinfection_probability = r4, isolated = 0.8)

data4, _ = run!(sir_model4, agent_step!, model_step!, 500; adata)

l4 = lines!(ax, data4[:, dataname((:status, infected))], color = :red)
figure[1, 2] = Legend(
    figure,
    [l1, l2, l3, l4],
    ["r=$r1", "r=$r2", "r=$r1", "r=$r4, social distancing"],
)

println("Już___________________________________________________________________")
figure