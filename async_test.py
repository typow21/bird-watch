import requests
import aiohttp
import asyncio
import time
# run 10 requests to localhost:80 in parallel and print the time it took
async def fetch(session):
    async with session.get('http://localhost:80') as response:
        return await response.text()

async def sleep_func():
    time.sleep(5)
    return {"message": "Hello World"}

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session) for i in range(10)]
        responses = await asyncio.gather(*tasks)
        # for response in responses:
        #     print(response)

if __name__ == "__main__":
    total_start_time = time.time()
    tasks = []
    for x in range(10):
        start_time = time.time()
        tasks.append(main())
        end_time = time.time()
        print(end_time - start_time)
    asyncio.run(asyncio.wait(tasks))
    total_end_time = time.time()
    print(total_end_time - total_start_time)