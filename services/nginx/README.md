## Nginx

Nginx is a reverse proxy server which allows a single point of entry for multiple services. This is useful for managing multiple services on a single domain, and for managing the routing of requests to different services. In this project, Nginx is used to route requests to the Jupyter notebook and MLFlow services, such that the services can run on a higher-performance machine on the USNA domain, or on your local machine, with the same behavior.
