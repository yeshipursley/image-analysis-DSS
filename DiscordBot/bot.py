import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio

from Commands import Scribe



bot = commands.Bot(command_prefix=".")

@bot.event
async def on_ready():
    print('We have logged in as {0.user}'.format(bot))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

bot.add_cog(Scribe(bot))

bot.run('TOKEN')