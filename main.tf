terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 2.13.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

data "docker_image" "tschabot" {
  name = "tschabot:latest"
}

resource "docker_container" "tschabot" {
  image = data.docker_image.tschabot.id
  name  = "tschabot"
  ports {
    internal = 8501
    external = 8501
  }
}
