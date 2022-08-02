library(data.table)
library(ggplot2)

x <- function(file){
  data <- fread(file)
  
  data$row <- 1:nrow(data)
  data
  dt <- melt(data, id.vars = "row")
  dt
  ggplot(dt, aes(y=row, x=variable, color=as.factor(value)))+
    geom_point(size=0.75, shape=15)+
    scale_color_manual(values = c("0"  = "#30358f",
                                  "1"  = "#5a5a5a",
                                  "2"  = "#ffffff",
                                  "3"  = "#bfbfbf",
                                  "4"  = "#ccc0da",
                                  "5"  = "#60497b",
                                  "6"  = "#953735",
                                  "7"  = "#948b54",
                                  "8"  = "#9db195",
                                  "9"  = "#f2dddc",
                                  "10" = "#dbeef3",
                                  "11" = "#93cddd",
                                  "12" = "#31849b",
                                  "13" = "#ff5050",
                                  "14" = "#ff9900",
                                  "15" = "#cccc00",
                                  "16" = "#fcd5b4",
                                  "17" = "#d99795",
                                  "18" = "#d7e4bc",
                                  "19" = "#66ff66",
                                  "20" = "#00b050",
                                  "21" = "#00823b",
                                  "22" = "#ffff00",
                                  "23" = "#fffbc1",
                                  "24" = "#ccff33",
                                  "25" = "#75923c",
                                  "26" = "#4f6228" 
    ))+
    theme_void()+
    theme(legend.position = "none")
}
y <- function(file){
  data <- fread(file)
  
  data$row <- 1:nrow(data)
  data
  dt <- melt(data, id.vars = "row")
  dt
  
  cityfile <- paste0("city",file)
  city <- fread(cityfile)
  city$row <- 1:nrow(city)
  city <- melt(city, id.vars = "row")
  city$value <- ifelse(city$value == 0, NA, city$value)
  city <- na.omit(city)
  
  roadfile <- paste0("road",file)
  road <- fread(roadfile)
  road$row <- 1:nrow(road)
  road <- melt(road, id.vars = "row")
  road$value <- ifelse(road$value == 0, NA, road$value)
  road <- na.omit(road)
  
  
  #print(summary(city$value))
  
  g <- ggplot(dt, aes(y=row, x=variable, fill=as.factor(value)))
  g <- g + geom_raster()
  g <- g + scale_fill_manual(values = c("0"  = "#8fd3ff",
                                  "1"  = "#eaaded",
                                  "2"  = "#a884f3",
                                  "3"  = "#905ea9",
                                  "4"  = "#6b3e75",
                                  "5"  = "#45293f",
                                  "6"  = "#6e2727",
                                  "7"  = "#547e64",
                                  "8"  = "#374e4a",
                                  "9"  = "#cddf6c",
                                  "10" = "#91db69",
                                  "11" = "#1ebc73",
                                  "12" = "#165a4c",
                                  "13" = "#b33831",
                                  "14" = "#ea4f36",
                                  "15" = "#b2ba90",
                                  "16" = "#fbff86",
                                  "17" = "#239063",
                                  "18" = "#d5e04b",
                                  "19" = "#a2a947",
                                  "20" = "#676633",
                                  "21" = "#4c3e24",
                                  "22" = "#f79617",
                                  "23" = "#30e1b9",
                                  "24" = "#0eaf9b",
                                  "25" = "#0b8a8f",
                                  "26" = "#0b5e65"   
    ),labels = c("0"  = "Ocean",
                 "1"  = "Polar Desert",
                 "2"  = "Ice Cap",
                 "3"  = "Tundra",
                 "4"  = "Wet Tundra",
                 "5"  = "Polar Wetlands",
                 "6"  = "Cool Desert",
                 "7"  = "Steppe",
                 "8"  = "Boreal Forest",
                 "9"  = "Temperate Woodlands",
                 "10" = "Temperate Forest",
                 "11" = "Temperate Wet Forest",
                 "12" = "Temperate Wetlands",
                 "13" = "Extreme Desert",
                 "14" = "Desert",
                 "15" = "Subtropical Scrub",
                 "16" = "Subtropical Woodlands",
                 "17" = "Mediterranean",
                 "18" = "Subtropical Dry Forest",
                 "19" = "Subtropical Forest",
                 "20" = "Subtropical Wet Forest",
                 "21" = "Suptropical Wetlands",
                 "22" = "Tropical Scrub",
                 "23" = "Tropical Woodlands",
                 "24" = "Tropical Dry Forest",
                 "25" = "Tropical Wet Forest",
                 "26" = "Tropical Wetlands"))+
    theme_void()+
    theme(legend.position = "right")+
    labs(fill="Biome", size="City Size")+
    scale_y_reverse()
  g <- g + geom_tile(data=road, aes(y=row, x=variable), fill="white")
  g <- g + geom_point(data=city, shape=20, color="black", fill="black", aes(size=value))
  g
  }

#x("map_m3.csv")
y("map_m3.csv")

