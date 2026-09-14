from discord.ext import commands


class GeneralCog(commands.Cog, name="General"):
    """General commands for the Discord bot."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.command(name="clear_memory")
    async def clear_memory(self, ctx: commands.Context) -> None:
        """Clear the conversation history in the current channel."""
        guild_id = ctx.guild.id if ctx.guild else None
        memory_manager = getattr(self.bot, "memory_manager", None)

        if memory_manager is not None:
            if memory_manager.clear(guild_id, ctx.channel.id):
                await ctx.send("🧹 Memory for this channel has been wiped clean!")
            else:
                await ctx.send("Memory was already empty here.")
        else:
            await ctx.send("⚠️ Memory manager is not configured.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(GeneralCog(bot))
